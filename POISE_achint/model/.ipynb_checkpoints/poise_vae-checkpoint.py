import torch
import torch.nn as nn
from .loss_calculator import Loss_Calculator
from .sampling import Sampling
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class POISE_VAE(nn.Module):
    def __init__(self, encoders, decoders, latent_dims=None, batch_size=-1,generate_mode=False):
        super(POISE_VAE,self).__init__()
        self.batch_size = batch_size
        self.latent_dims = tuple(map(lambda l: l.latent_dim, encoders))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.M = len(self.encoders)
        self.loss_calc = Loss_Calculator(self.M)
        self.sampler = Sampling(self.M)
    def encode(self, x):
        param1, param2 = [], []
        for i,xi in enumerate(x):
            if xi is None:
                ret = torch.zeros(self.batch_size, self.latent_dims[i]).to(device)
                param1.append(ret)
                param2.append(ret)  
            else:
                ret = self.encoders[i](xi)
                param1.append(ret[0])  ## list stores mu, param1 =[mu1,mu2]
                param2.append(ret[1])  ## list stores log_var, param2 = [log_var1, log_var2]
        return param1, param2
    def decode(self, z, x, generate_mode):
        param = []
        for i,xi in enumerate(z):
            ret = self.decoders[i](xi,x[i], generate_mode)
            param.append(ret)
        return param
    def forward(self, x, generate_mode):
        param1, param2 = self.encode(x)
        z_posteriors = self.sampler.vanilla_sampling(param1, param2)
        x_rec  = self.decode(z_posteriors, x, generate_mode)
        total_loss,rec_loss,kl_loss = self.loss_calc.loss(x, x_rec, param1, param2, generate_mode)
 
        results = {'data':x,'x_rec': x_rec,
            'total_loss': total_loss, 'rec_losses': rec_loss, 'KL_loss': kl_loss}
        return results
