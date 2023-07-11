"""Compression modeule for sbi.model"""

import torch
import numpy as np

class Compressor:
    """Compressor class for sbi.model"""
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, x):
        """Compresses the input data x using the model"""
        x = np.array(x)
    
        x = x[:, 2*30:, 11:14]
        x = x-x[:, 0, 2][:, np.newaxis, np.newaxis]
        x = torch.from_numpy(x).float().to(self.device)
        x = x.flatten(start_dim=1)

        return x

    def __repr__(self):
        return f'Compressor(device={self.device})'