import os
import copy
import random
import argparse
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from INFORMS23_UTILS import CustomDataset, CosineAnnealingWarmUpRestarts, EarlyStopping, str2bool, he_init_normal, pot, dspot
from INFORMS23_MODEL import C_AutoEncoder, Deep_SVDD

import warnings
warnings.filterwarnings("ignore")


class TrainerDeepSVDD:
    def __init__(self, config):
        self.config = config
        self.R = torch.tensor([0], device=config['device'])
        self.c = None
    
    def pretrain(self):
        ae = C_AutoEncoder(self.config).to(self.config['device'])
        ae.apply(he_init_normal)
        ae = nn.DataParallel(ae).to(self.config['device'])
            
        optimizer = torch.optim.AdamW(ae.parameters(), lr=self.config['lr_ae'], weight_decay=self.config['weight_decay_ae'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
        ae.train()
        for epoch in range(self.config['num_epochs_ae']):
            epoch_num = epoch + 1
            
            train_dataset = CustomDataset(self.config['train_dataset'])
            train_loader=torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=self.config['batch_size'], 
                                                    shuffle=False, 
                                                    drop_last=False)
            total_loss = 0
            
            for train_data in train_loader:
                x = train_data.to(self.config['device']) 
    
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.8f}'.format(
                    epoch_num, total_loss/len(train_loader)))
            
            self.save_weights_for_DeepSVDD(ae, train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        self.c = self.set_c(model, dataloader)
        model = Deep_SVDD(self.config).to(self.config['device'])
        model = nn.DataParallel(model).to(self.config['device'])
        
        state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        model.module.load_state_dict(state_dict, strict=False) if torch.cuda.device_count() > 1 else model.load_state_dict(state_dict, strict=False)

        torch.save({'center': self.c.cpu().data.numpy().tolist(),
                    'net_dict': model.state_dict()}, '{}_pretrained_parameters.pt'.format(self.config['model_save_name']))
    

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.config['device']) 
                z = model.module.encoder(x) if isinstance(model, nn.DataParallel) else model.encoder(x)

                z_.append(z.detach())
        z_ = torch.cat(z_)
        pretrained_center = torch.mean(z_, dim=0)
        pretrained_center[(abs(pretrained_center) < eps) & (pretrained_center < 0)] = -eps
        pretrained_center[(abs(pretrained_center) < eps) & (pretrained_center > 0)] = eps
        return pretrained_center
    
    
    def get_radius(self, dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


    def kl_divergence(self, q, p, dim=0):
        # Compute KL divergence
        kl_div = torch.sum(p * (torch.log(p) - torch.log(q)), dim=dim)
        return kl_div


    def train(self):
        
        train_dataset = CustomDataset(self.config['train_dataset'])
        train_loader=torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.config['batch_size'], 
                                                shuffle=False, 
                                                drop_last=False)
    
        
        """Training the Deep SVDD model"""
        self.config['mode'] = 'train'
        model = Deep_SVDD(self.config).to(self.config['device'])
        model = nn.DataParallel(model).to(self.config['device'])
        
        if self.config['pretrain']==True:
            state_dict = torch.load('{}_pretrained_parameters.pt'.format(self.config['model_save_name']))
            model.module.load_state_dict(state_dict['net_dict'], strict=False) if torch.cuda.device_count() > 1 else model.load_state_dict(state_dict['net_dict'], strict=False)
            self.c = torch.Tensor(state_dict['center']).to(self.config['device'])
        else:
            model.apply(he_init_normal)
            self.c = model.module.Centroid.weight.data[0]
        
        if self.config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['lr_t'], eta_min=0)
            
        elif self.config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=self.config['weight_decay'])
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.config['lr_t'], eta_max=self.config['lr'], gamma=self.config['gamma'], T_mult=1, T_up=0)
        
        early_stopping = EarlyStopping(patience=self.config['patience'], mode='min', min_delta=self.config['min_delta'])
        
        if self.config['loss_function'] == 'BCE':
            loss_function = nn.BCELoss()
            
        elif self.config['loss_function'] == 'KL':
            loss_function = self.kl_divergence
            
        best_loss=100
        Total_Loss_Dict = {
            'mapping_loss': [],
            'clustering_loss': [],
            'total_loss': []
        }
        
        model.train()
        for epoch in range(self.config['num_epochs']):
            
            epoch_num = epoch + 1
            mapping_loss_epoch = 0
            ADEC_loss_epoch = 0
            total_loss = 0

            for train_data in train_loader:
                x = train_data.to(self.config['device']) 

                optimizer.zero_grad()
                
                if self.config['model_name'] == 'DeepSVDD':
                    p, q, z, _ = model(x)
                    ADEC_loss = torch.tensor([0], device=self.config['device']) 
                                        
                elif self.config['model_name'] == 'ADEC':
                    p, q, z, c = model(x)
                    ADEC_loss = loss_function(q, p)
                                        
                    # When you use nn.DataParallel, then the output's batch size will be increased.
                    if (c.shape[0] > 1): 
                        self.c = c[0]
                
                dist = torch.sum((z - self.c) ** 2, dim=1)

                if self.config['objective'] == 'soft-boundary':
                    scores = dist - self.R ** 2
                    mapping_loss = self.R ** 2 + (1 / self.config['nu']) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    mapping_loss = torch.mean(dist)
                    
                if self.config['loss_type'] == 'weighted':
                    loss = (1/epoch_num * mapping_loss) + (1-1/epoch_num * ADEC_loss)
                else:
                    loss = mapping_loss + (self.config['lambda_param'] * ADEC_loss)
                    
                loss.backward()
                optimizer.step()
                
                # Update hypersphere radius R on mini-batch distances
                if (self.config['objective'] == 'soft-boundary'):
                    with torch.no_grad():
                        self.R.copy_(torch.tensor(self.get_radius(dist, self.config['nu']), device=self.config['device']))

                mapping_loss_epoch += mapping_loss.item()
                ADEC_loss_epoch += ADEC_loss.item()
                total_loss += loss.item()
            
            Total_Loss_Dict['mapping_loss'].append(mapping_loss_epoch/len(train_loader))
            Total_Loss_Dict['clustering_loss'].append(ADEC_loss_epoch/len(train_loader))
            Total_Loss_Dict['total_loss'].append(total_loss/len(train_loader))
                        
            scheduler.step()
            
            print_best = 0    
            if Total_Loss_Dict['total_loss'][-1] <= best_loss:
                difference = best_loss - Total_Loss_Dict['total_loss'][-1]
                best_loss = Total_Loss_Dict['total_loss'][-1]
                        
                best_idx = epoch_num
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save({'center': self.c.cpu().data.numpy().tolist(),
                            'radius': self.R.cpu().data.numpy().tolist(),
                            'net_dict': model.state_dict()}, 
                            '{}.pt'.format(self.config['model_save_name']))  
                
                print_best = "==> %dEpoch Model Saved  |  Best Loss: %.8f  |  Difference %.8f"%(best_idx, best_loss, difference)

            print('Epoch: {} ,    Total Loss: {:.9f},    Mapping Loss: {:.9f},    Clustering Loss: {:.9f},    Best Loss: {:.9f}'.format(
                    epoch_num, Total_Loss_Dict['total_loss'][-1], Total_Loss_Dict['mapping_loss'][-1], Total_Loss_Dict['clustering_loss'][-1], best_loss))
            None if type(print_best)==int else print(print_best,'\n')

            if early_stopping.step(torch.tensor(Total_Loss_Dict['total_loss'][-1])):
                break
            

    def test(self):
    
        test_dataset = CustomDataset(self.config['test_dataset'])
        test_loader=torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=self.config['batch_size'], 
                                                shuffle=False, 
                                                drop_last=False)
        
        """Training the Deep SVDD model"""
        self.config['mode'] = 'test'
        model = Deep_SVDD(self.config).to(self.config['device'])
        model = nn.DataParallel(model).to(self.config['device'])
        
        state_dict = torch.load('{}_pretrained_parameters.pt'.format(self.config['model_save_name']))
        model.module.load_state_dict(state_dict['net_dict'], strict=False) if torch.cuda.device_count() > 1 else model.load_state_dict(state_dict['net_dict'], strict=False)
        self.c = torch.Tensor(state_dict['center']).to(self.config['device'])
        self.R = torch.Tensor(state_dict['radius']).to(self.config['device'])

        if self.config['loss_function'] == 'BCE':
            loss_function = nn.BCELoss()
            
        elif self.config['loss_function'] == 'KL':
            loss_function = self.kl_divergence
        
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for test_inputs in test_loader:
                test_x = test_inputs.to(self.config['device']) 
                
                p, q, z = model(test_x)
                
                if self.config['model_name'] == 'DeepSVDD':
                    ADEC_loss = torch.tensor([0], device=self.config['device']) 
                                        
                elif self.config['model_name'] == 'ADEC':
                    ADEC_loss = loss_function(q, p)
                
                
                
                
                dim 계산 해야됨 
                dist = torch.sum((z - self.c) ** 2, dim=1)

                if self.config['objective'] == 'soft-boundary':
                    scores = dist - self.R ** 2
                    
                    
                    
                    여기도 dim rㅖ산해서 넣어야함
                    mapping_loss = self.R ** 2 + (1 / self.config['nu']) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    mapping_loss = torch.mean(dist)
                
                Annomaly_score = mapping_loss + (self.config['lambda_param'] * ADEC_loss)
                
                preds = torch.sigmoid(outputs).round()  # Assuming binary classification
                predictions.extend(preds)

        predictions = torch.stack(predictions).squeeze().numpy()



        # Convert to a format that's easy to handle (like numpy)
        predicted_labels_np = predicted_labels.numpy()
        


        # Parameters
        risk = 1e-4
        init_level = 0.98
        num_candidates = 10
        epsilon = 1e-8


        # Test
        z, t = pot(torch.tensor(data).detach().cpu().numpy(), risk, init_level, num_candidates, epsilon)


    # def save_model(self, export_model, save_ae=True):
    #     """Save Deep SVDD model to export_model."""

    #     net_dict = self.model.state_dict()
    #     ae_net_dict = self.ae_net.state_dict() if save_ae else None

    #     torch.save({'R': self.R,
    #                 'c': self.c,
    #                 'net_dict': net_dict,
    #                 'ae_net_dict': ae_net_dict}, export_model)

    # def load_model(self, model_path, load_ae=False):
    #     """Load Deep SVDD model from model_path."""

    #     model_dict = torch.load(model_path)

    #     self.R = model_dict['R']
    #     self.c = model_dict['c']
    #     self.model.load_state_dict(model_dict['net_dict'])
    #     if load_ae:
    #         if self.ae_net is None:
    #             self.ae_net = build_autoencoder(self.net_name)
    #         self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    # def save_results(self, export_json):
    #     """Save results dict to a JSON-file."""
    #     with open(export_json, 'w') as fp:
    #         json.dump(self.results, fp)


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)
    
    
    # Model parameters
    parser.add_argument('--model_name', default='DeepSVDD', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--latent_vector', default=32, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--drop_out', default=0.1, type=float)
    parser.add_argument('--nu', default=0.1, type=float)
    parser.add_argument('--lambda_param', default=0.5, type=float)
    parser.add_argument('--objective', default='soft-boundary', type=str)
    parser.add_argument('--loss_type', default='mean', type=str)
    parser.add_argument('--loss_function', default='BCE', type=str)


    # Optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_ae', default=1e-3, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--min_delta', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--weight_decay_ae', default=0.0001, type=float)


    # Training parameters
    parser.add_argument('--train_path', default='/home/sy.lee-home/sylee/OKESTRO/INFORMS-Anomaly-Detection/dataset/SMAP/SMAP_train.npy', type=str)
    parser.add_argument('--test_path', default='/home/sy.lee-home/sylee/OKESTRO/INFORMS-Anomaly-Detection/dataset/SMAP/SMAP_test.npy', type=str)
    parser.add_argument('--pretrain', default=True, type=str2bool)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_epochs_ae', default=100, type=int)
    parser.add_argument('--warm_up_n_epochs', default=30, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):
    
    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'latent_vector': args.latent_vector,
        'depth': args.depth,
        'drop_out': args.drop_out,
        'nu': args.nu,
        'lambda_param': args.lambda_param,
        'objective': args.objective,
        'loss_type': args.loss_type,
        'loss_function': args.loss_function,
        
        # Optimizer parameters
        'lr': args.lr,
        'lr_ae': args.lr_ae,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'weight_decay': args.weight_decay,
        'weight_decay_ae': args.weight_decay_ae,
        
        # Training parameters
        'train_path': args.train_path,
        'test_path': args.test_path,
        'pretrain' : args.pretrain,
        'num_epochs': args.num_epochs,
        'num_epochs_ae': args.num_epochs_ae,
        'warm_up_n_epochs': args.warm_up_n_epochs,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device,
        }
    
    assert config['objective'] in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
    assert (0 < config['nu']) & (config['nu'] <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
    assert config['model_name'] in ('DeepSVDD', 'ADEC'), "Model must be either 'DeepSVDD' or 'ADEC'."
    assert config['loss_type'] in ('mean', 'weighted'), "Calcualtion must be either 'mean' or 'weighted'."
    assert config['loss_function'] in ('KL', 'BCE'), "Loss function must be either 'KL' or 'BCE'."

    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['model_name'])+"_"+\
                                                                str(config['batch_size'])+"_"+\
                                                                str(config['hidden_dim'])+"_"+\
                                                                str(config['latent_vector'])+"_"+\
                                                                str(config['depth'])+"_"+\
                                                                str(config['drop_out'])+"_"+\
                                                                str(config['nu'])+"_"+\
                                                                str(config['lambda_param'])+"_"+\
                                                                str(config['objective'])+"_"+\
                                                                str(config['loss_type'])+"_"+\
                                                                str(config['loss_function'])+"|"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_ae'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['min_delta'])+"_"+\
                                                                str(config['weight_decay'])+"_"+\
                                                                str(config['weight_decay_ae'])+"|"+\
                                                                str(config['num_epochs'])+"_"+\
                                                                str(config['num_epochs_ae'])+"_"+\
                                                                str(config['warm_up_n_epochs'])
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
    # -------------------------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device

    # -------------------------------------------------------------------------------------------

    # Dataload
    
    config['train_dataset'] = np.load(config['train_path'])
    config['test_dataset'] = np.load(config['test_path'])
    config['test_label'] = np.load(config['test_path'].split('.npy')[0] + '_label.npy')
    config['input_dim'] = config['train_dataset'].shape[1]
    config['mode'] = 'TRAIN'
    
    # -------------------------------------------------------------------------------------------
    
    deep_SVDD = TrainerDeepSVDD(config)

    if config['pretrain']:
        deep_SVDD.pretrain()
    deep_SVDD.train()
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
