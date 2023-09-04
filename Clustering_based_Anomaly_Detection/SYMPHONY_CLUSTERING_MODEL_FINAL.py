import torch 
import torch.nn as nn


class Deep_SVDD(nn.Module):
    def __init__(self, config):
        super(Deep_SVDD, self).__init__()
        self.conv1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.conv2 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2))
        self.conv3 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/4))
        self.conv4 = nn.Linear(int(config['hidden_dim']/4), int(config['hidden_dim']/8))
        self.conv5 = nn.Linear(int(config['hidden_dim']/8), config['latent_vector'])
        
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
        self.drop_out = nn.Dropout(config['drop_out'])
        self.Centroid = nn.Linear(config['latent_vector'], 1, bias=False)
        self.mode = config['mode']
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv5(x)
    
        q = self.sigmoid(self.Centroid(x))
        p =(q >= 0.5).float()
        
        if self.mode == 'train':
            return p, q, x, self.Centroid.weight
        else:
            return p, q, x
    
class C_AutoEncoder(nn.Module):
    def __init__(self, config):
        super(C_AutoEncoder, self).__init__()

        self.conv1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.conv2 = nn.Linear(config['hidden_dim'], int(config['hidden_dim']/2))
        self.conv3 = nn.Linear(int(config['hidden_dim']/2), int(config['hidden_dim']/4))
        self.conv4 = nn.Linear(int(config['hidden_dim']/4), int(config['hidden_dim']/8))
        self.conv5 = nn.Linear(int(config['hidden_dim']/8), config['latent_vector'])
        
        self.deconv1 = nn.Linear(config['latent_vector'], int(config['hidden_dim']/8))
        self.deconv2 = nn.Linear(int(config['hidden_dim']/8), int(config['hidden_dim']/4))
        self.deconv3 = nn.Linear(int(config['hidden_dim']/4), int(config['hidden_dim']/2))
        self.deconv4 = nn.Linear(int(config['hidden_dim']/2), config['hidden_dim'])
        self.deconv5 = nn.Linear(config['hidden_dim'], config['input_dim'])
        
        self.drop_out = nn.Dropout(config['drop_out'])
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        x = self.conv1(x)        
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv5(x)

        return x
    
    def decoder(self, z):
        z = self.deconv1(z)
        z = self.relu(z)
        z = self.drop_out(z)
        z = self.deconv2(z)
        z = self.relu(z)
        z = self.drop_out(z)
        z = self.deconv3(z)
        z = self.relu(z)
        z = self.drop_out(z)
        z = self.deconv4(z)
        z = self.relu(z)
        z = self.drop_out(z)
        z = self.deconv5(z)
        
        return z
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        return x_hat

