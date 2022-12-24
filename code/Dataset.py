import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FootBall_Dataset(Dataset):

    def __init__(self, x, y, device='cpu'):
        self.n_sample = x.shape[0]
        self.x_data = torch.as_tensor(np.array(x).astype(np.float32)).to(device)
        self.y_data = torch.as_tensor(np.array(y).astype(np.float32)).to(device)

    
    def __getitem__(self, index):

        x = self.x_data[index]
        y = self.y_data[index]

        return x, y
    

    def __len__(self): 
        return self.n_sample

