import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampling:
    def __init__(self, M):
        self.M = M
    def vanilla_sampling(self, mu,log_var):
        z_posterior = []
        for i in range(self.M):
            std = torch.exp(0.5*log_var[i])
            eps = torch.randn_like(std)
            z_posterior.append(mu[i]+std*eps)
        return z_posterior