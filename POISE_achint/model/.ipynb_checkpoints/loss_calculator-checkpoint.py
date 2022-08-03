import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loss_Calculator:
    def __init__(self, M):
        self.M = M
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.color_level = 256
    def rec_loss_calc(self, x, x_rec, generate_mode):
        loss = torch.tensor(0.0).to(device)
        for i in range(self.M):
            if x[i] is None:
                loss += torch.tensor(0.0).to(device)
            elif generate_mode is False:
                x1 = (x[i] * (self.color_level - 1)).floor().long()
                x1_rec = x_rec[i].flatten(-3, -1)
                loss +=self.ce_loss(x1_rec, x1)
            else:
                x_rec[i] = x_rec[i].flatten(-3,-1)
                loss +=self.mse_loss(x[i], x_rec[i])
        return loss
    def kl_loss_calc(self,mu,log_var):
        loss = 0
        for i in range(self.M):
            var = torch.exp(log_var[i])
            loss +=torch.sum(-0.5*(1+log_var[i]-mu[i]**2-var))
        return loss
    def loss(self,x, x_rec, mu, log_var, generate_mode):

        rec_loss = self.rec_loss_calc(x, x_rec, generate_mode)

        kl_loss  = self.kl_loss_calc(mu,log_var)
        total_loss = rec_loss+kl_loss
        return total_loss,rec_loss,kl_loss