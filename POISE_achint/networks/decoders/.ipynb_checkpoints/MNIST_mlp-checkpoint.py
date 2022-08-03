import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DecMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(DecMNIST, self).__init__()  
        self.latent_dim = latent_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 128), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 512), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, self.dim_MNIST), 
                                 nn.Sigmoid())        
    def forward(self, z, x):
        return self.dec(z).to(device)