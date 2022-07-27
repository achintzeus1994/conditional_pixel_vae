import torch
import scipy.io as sio
import random
import numpy as np

class MNIST_MNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path_1, mnist_pt_path_2):

        self.mnist_pt_path_1 = mnist_pt_path_1
        self.mnist_pt_path_2 = mnist_pt_path_2
            
        # Load the pt for MNIST 
        self.mnist_data_1, self.mnist_targets_1 = torch.load(self.mnist_pt_path_1)
        
        # Load the pt for MNIST 

        self.mnist_data_2, self.mnist_targets_2 = torch.load(self.mnist_pt_path_2)

        
        
    def __len__(self):
        return len(self.mnist_data_1)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        mnist_img_1, mnist_target_1 = self.mnist_data_1[index], int(self.mnist_targets_1[index])
        
        # Randomly pick an index from the indices list
        mnist_img_2, mnist_target_2 = self.mnist_data_2[index], int(self.mnist_targets_2[index])
        
        # unsqueeze to add channel dim
        return mnist_img_1.unsqueeze(0)/255, mnist_img_2.unsqueeze(0)/255, mnist_target_1, mnist_target_2
