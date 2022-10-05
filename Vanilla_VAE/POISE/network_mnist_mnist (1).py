import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EncMNIST(nn.Module):
    def __init__(self, latent_dim_mnist, num_mix):
        super(EncMNIST, self).__init__()
        self.latent_dim_mnist = latent_dim_mnist
        self.dim_MNIST = 28 * 28
        self.K =num_mix
        self.enc_1 = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(), 
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.K) )
        self.enc_2 = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(), 
                                 nn.Linear(512, 128),
                                 nn.ReLU())
        
        post_mean = []
        post_var  = []
        for i in range(self.K):
            post_mean.append(nn.Linear(128, latent_dim_mnist))
            post_var.append(nn.Linear(128, latent_dim_mnist))
        self.post_mean = nn.ParameterList(post_mean)
        self.post_var = nn.ParameterList(post_var)
#         self.enc_mu_mnist = nn.Linear(128, latent_dim_mnist)
#         self.enc_var_mnist = nn.Linear(128, latent_dim_mnist)
#     def sample_gumbel(self,shape, eps=1e-20):
#         U = torch.rand(shape).to(device)
#         return -torch.log(-torch.log(U + eps) + eps)


#     def gumbel_softmax_sample(self,logits, temperature):
#         y = logits + self.sample_gumbel(logits.size())
#         return F.softmax(y / temperature, dim=-1)
#     def gumbel_softmax(self,logits, temperature, hard=False):
#         """
#         ST-gumple-softmax
#         input: [*, n_class]
#         return: flatten --> [*, n_class] an one-hot vector
#         """
#         y = self.gumbel_softmax_sample(logits, temperature)
#         if not hard:
#             return y.view(-1, self.K)

#         shape = y.size()
#         _, ind = y.max(dim=-1)
#         y_hard = torch.zeros_like(y).view(-1, shape[-1])
#         y_hard.scatter_(1, ind.view(-1, 1), 1)
#         y_hard = y_hard.view(*shape)
#         # Set gradients w.r.t. y_hard gradients w.r.t. y
#         y_hard = (y_hard - y).detach() + y
#         return y_hard.view(-1, self.latent_dim_mnist * self.K)

    def forward(self, x):
        phi = self.enc_1(x)
#         y = self.gumbel_softmax(enc_1_out, 1)
        enc_2_out = self.enc_2(x)
        mean = []
        log_var = []
        for i in range(self.K):
            mean.append(self.post_mean[i](enc_2_out))
            log_var.append(self.post_var[i](enc_2_out))
        return mean, log_var, phi
    
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
#         return self.dec(z).reshape(-1, 1, 28, 28).to(z.device)
        return self.dec(z).to(device)