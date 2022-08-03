import torch
import torch.nn as nn

class EncMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(EncMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu_mnist = nn.Linear(128, latent_dim)
        self.enc_var_mnist = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        mu_mnist = self.enc_mu_mnist(x)
        log_var_mnist = self.enc_var_mnist(x)
        return mu_mnist, log_var_mnist
    
