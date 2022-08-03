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
    

class DecMNIST(nn.Module):
    def __init__(self, pixelcnn, color_level):
        super(DecMNIST, self).__init__()
        self.pixelcnn = pixelcnn
        self.color_level = color_level
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, z, x, generate_mode):
        if x is not None:
            x = x.reshape(z.shape[0], 1, 28, 28).to(z.device)
        else:
            x = torch.zeros(z.shape[0], 1, 28, 28).to(z.device)
            
        if generate_mode is False:
            sample = self.pixelcnn(x, z)
            x = (x.flatten(-3, -1) * (self.color_level - 1)).floor().long()
            return sample, self.ce_loss(sample.flatten(-3, -1), x)
        else:
            shape = [1,28,28]
            sample = self.pixelcnn.sample(x, shape, z, x.device)
            sample = torch.exp(sample)
            return sample, self.mse_loss(sample, x)
            
            
class EncSVHN(nn.Module):
    def __init__(self, latent_dim):
        super(EncSVHN, self).__init__()
        self.latent_dim = latent_dim
        
        n_channels = (3, 32, 64, 128)
        kernels = (4, 4, 4)
        strides = (2, 2, 2)
        paddings = (1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Conv2d(in_channels=128, out_channels=latent_dim, 
                                kernel_size=4, stride=1, padding=0)
        self.enc_var = nn.Conv2d(in_channels=128, out_channels=latent_dim, 
                                 kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.enc(x)
        # Be careful not to squeeze the batch dimension if batch size = 1
        mu = self.enc_mu(x).squeeze(3).squeeze(2)
        log_var = self.enc_var(x).squeeze(3).squeeze(2)
        return mu, log_var
        
class DecSVHN(nn.Module):
    def __init__(self, pixelcnn, color_level):
        super(DecSVHN, self).__init__()  
        self.pixelcnn = pixelcnn
        self.color_level = color_level
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, z, x, generate_mode):
        if x is not None:
            x = x.reshape(z.shape[0], 3, 32, 32).to(z.device)
        else:
            x = torch.zeros(z.shape[0], 3, 32, 32).to(z.device)
            
        if generate_mode is False:
            sample = self.pixelcnn(x, z)
            x = (x.flatten(-3, -1) * (self.color_level - 1)).floor().long()
            return sample, self.ce_loss(sample.flatten(-3, -1), x)
        else:
            shape = [3,32,32]
            sample = self.pixelcnn.sample(x, shape, z, x.device)
            sample = torch.exp(sample)
            return sample, self.mse_loss(sample, x)