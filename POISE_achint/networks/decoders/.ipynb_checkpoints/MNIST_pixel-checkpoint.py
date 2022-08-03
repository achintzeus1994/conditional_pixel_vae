import torch
import torch.nn as nn

class DecMNIST(nn.Module):
    def __init__(self, pixelcnn, color_level=256):
        super(DecMNIST, self).__init__()
        self.pixelcnn = pixelcnn
        self.color_level = color_level
        
    def forward(self, z, x, generate_mode):
        if x is None:
            x = torch.zeros(z.shape[0], 1, 28, 28).to(z.device)            
        else:
            x = x.reshape(z.shape[0], 1, 28, 28).to(z.device)
        if generate_mode is False:
            sample = self.pixelcnn(x, z)
            return sample 
        else:
            shape = [1,28,28]
            sample = self.pixelcnn.sample(x, shape, z, x.device)
            return sample