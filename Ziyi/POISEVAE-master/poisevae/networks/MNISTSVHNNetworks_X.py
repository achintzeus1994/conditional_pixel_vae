import torch
import torch.nn as nn

class EncMNIST(nn.Module):
    def __init__(self, latent_dim_mnist, latent_dim_svhn):
        super(EncMNIST, self).__init__()
        self.latent_dim_mnist = latent_dim_mnist
        self.latent_dim_svhn = latent_dim_svhn
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu_mnist = nn.Linear(128, latent_dim_mnist)
        self.enc_var_mnist = nn.Linear(128, latent_dim_mnist)
        self.enc_mu_svhn = nn.Linear(128, latent_dim_svhn)
        self.enc_var_svhn = nn.Linear(128, latent_dim_svhn)

    def forward(self, x):
        x = self.enc(x.flatten(-2, -1)) # assume x is originally 28 * 28
        mu_mnist = self.enc_mu_mnist(x)
        log_var_mnist = self.enc_var_mnist(x)
        mu_svhn = self.enc_mu_svhn(x)
        log_var_svhn = self.enc_var_svhn(x)
        return mu_mnist, log_var_mnist, mu_svhn, log_var_svhn

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
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)
    
class EncSVHN(nn.Module):
    def __init__(self, latent_dim_svhn, latent_dim_mnist):
        super(EncSVHN, self).__init__()
        self.latent_dim_mnist = latent_dim_mnist
        self.latent_dim_svhn = latent_dim_svhn
        
        n_channels = (3, 32, 64, 128)
        kernels = (4, 4, 4)
        strides = (2, 2, 2)
        paddings = (1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu_svhn = nn.Conv2d(in_channels=128, out_channels=latent_dim_svhn, 
                                     kernel_size=4, stride=1, padding=0)
        self.enc_var_svhn = nn.Conv2d(in_channels=128, out_channels=latent_dim_svhn, 
                                      kernel_size=4, stride=1, padding=0)
        self.enc_mu_mnist = nn.Conv2d(in_channels=128, out_channels=latent_dim_mnist, 
                                      kernel_size=4, stride=1, padding=0)
        self.enc_var_mnist = nn.Conv2d(in_channels=128, out_channels=latent_dim_mnist, 
                                       kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.enc(x)
        # Be careful not to squeeze the batch dimension if batch size = 1
        mu_svhn = self.enc_mu_svhn(x).squeeze(3).squeeze(2)
        log_var_svhn = self.enc_var_svhn(x).squeeze(3).squeeze(2)
        mu_mnist = self.enc_mu_mnist(x).squeeze(3).squeeze(2)
        log_var_mnist = self.enc_var_mnist(x).squeeze(3).squeeze(2)
        return mu_svhn, log_var_svhn, mu_mnist, log_var_mnist
    
class DecSVHN(nn.Module):
    def __init__(self, latent_dim):
        super(DecSVHN, self).__init__()  
        self.latent_dim = latent_dim
        
        n_channels = (latent_dim, 128, 64, 32, 3)
        kernels = (4, 4, 4, 4)
        strides = (1, 2, 2, 2)
        paddings = (0, 1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.ConvTranspose2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
        li[-1] = nn.Sigmoid()
        
        self.dec = nn.Sequential(*li)
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)