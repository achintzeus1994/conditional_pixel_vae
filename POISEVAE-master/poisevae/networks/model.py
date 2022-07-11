import torch
import torch.nn as nn
import torch.nn.functional as F
# from conv_layers import MaskedConv2d, CroppedConv2d
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

import numpy as np


class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)

        kernel_height, _ = self.kernel_size
        res = x[:, :, 1:-kernel_height, :]
        shifted_up_res = x[:, :, :-kernel_height-1, :]

        return res, shifted_up_res


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type, data_channels, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.zeros(self.weight.size(), dtype=np.float32)
        mask[:, :, :yc, :] = 1
        mask[:, :, yc, :xc + 1] = 1

        def cmask(out_c, in_c):
            a = (np.arange(out_channels) % data_channels == out_c)[:, None]
            b = (np.arange(in_channels) % data_channels == in_c)[None, :]
            return a * b

        for o in range(data_channels):
            for i in range(o + 1, data_channels):
                mask[cmask(o, i), yc, xc] = 0

        if mask_type == 'A':
            for c in range(data_channels):
                mask[cmask(c, c), yc, xc] = 0

        mask = torch.from_numpy(mask).float()

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels):
        super(CausalBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='A',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='A',
                                 data_channels=data_channels)

    def forward(self, image):
        v_out, v_shifted = self.v_conv(image)
        v_out += self.v_fc(image)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(image)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)
        h_out = self.h_fc(h_out)

        return v_out, h_out


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels, lat_dim):
        super(GatedBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = MaskedConv2d(2 * out_channels,
                                   2 * out_channels,
                                   (1, 1),
                                   mask_type='B',
                                   data_channels=data_channels)

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='B',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='B',
                                 data_channels=data_channels)

        self.h_skip = MaskedConv2d(out_channels,
                                   out_channels,
                                   (1, 1),
                                   mask_type='B',
                                   data_channels=data_channels)

#         self.latent_mapping = nn.Sequential(nn.Linear(lat_dim, 128), 
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(128, 512), 
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(512, 784), 
#                                  nn.Sigmoid())
        self.latent_mapping = nn.Sequential(nn.Linear(lat_dim, 8), nn.ReLU(inplace=True), 
                                            nn.Linear(8, 2*out_channels))

    def forward(self, x):
        v_in, h_in, skip, label = x[0], x[1], x[2], x[3]

        latent_embedded = self.latent_mapping(label).unsqueeze(2).unsqueeze(3)

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out += latent_embedded
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out += latent_embedded
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        # skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label}


class PixelCNN(nn.Module):
    def __init__(self, lat_dim, data_channels, causal_ksize=7, hidden_ksize=7, hidden_fmaps=30, out_hidden_fmaps=10, hidden_layers=4):
        """
        Parameters
        ----------
        causal_ksize: int, default=7
            Kernel size of causal convolution
            
        hidden_ksize: int, default=7
            Kernel size of hidden layers convolutions

        color_levels: int, default=2 (removed)
            Number of levels to quantisize value of each channel of each pixel into

        hidden_fmaps: int, default=30
            Number of feature maps in hidden layer (must be divisible by 3)
            
        out_hidden_fmaps: int, default=10
            Number of feature maps in outer hidden layer
            
        hidden_layers: int, default=4
            Number of layers of gated convolutions with mask of type "B"
        """
        super(PixelCNN, self).__init__()
        self.hidden_fmaps = hidden_fmaps
        self.color_levels = 256
        self.causal_conv = CausalBlock(data_channels,
                                       hidden_fmaps,
                                       causal_ksize,
                                       data_channels=data_channels)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(hidden_fmaps, hidden_fmaps, hidden_ksize, data_channels, lat_dim) for _ in range(hidden_layers)]
        )


#         self.latent_mapping = nn.Sequential(nn.Linear(lat_dim, 128), 
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(128, 512), 
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(512, 784), 
#                                  nn.Sigmoid())
        self.latent_mapping = nn.Sequential(nn.Linear(lat_dim, 8), nn.ReLU(inplace=True), 
                                            nn.Linear(8, self.hidden_fmaps))

        self.out_hidden_conv = MaskedConv2d(hidden_fmaps,
                                            out_hidden_fmaps,
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=data_channels)

        self.out_conv = MaskedConv2d(out_hidden_fmaps,
                                     data_channels*self.color_levels,
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=data_channels)

    def forward(self, image, label):
        count, data_channels, height, width = image.size()
#         print('image',image.size())
        v, h = self.causal_conv(image)   #v,h = (batchsize,30,28,28)    

        _, _, out, _ = self.hidden_conv({0: v,
                                         1: h,
                                         2: image.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True),
                                         3: label}).values()
#       out =[batch size, hidden_fmaps, 28, 28]
        latent_embedded = self.latent_mapping(label).unsqueeze(2).unsqueeze(3) ## latent_embedded = batch size x hidden_fmaps x 1 x 1 
        # add label bias
        out += latent_embedded  
        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)
        out = out.view(count,self.color_levels, data_channels, height, width)  # out =[batch size, 256, 28, 28]
        return (out)



    def sample(self,img, shape, count, label=None, device='cuda'):
        channels, height, width = shape

        samples = torch.zeros_like(img).to(device).clone()        
        labels = label.to(device)
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples, labels)  # unnormalized_probs= [batch size, 256, 28, 28]
#                         print(unnormalized_probs.mean(), unnormalized_probs.std(), unnormalized_probs.max(), unnormalized_probs.min())
                        pixel_probs = torch.softmax(unnormalized_probs[:,:, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze().float() / 255
                        samples[:, c, i, j] = sampled_levels
 
    
        return (samples)