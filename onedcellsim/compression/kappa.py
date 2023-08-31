import torch
import numpy as np

class Compressor:
    """Compressor class for sbi.model. This one includes kappa."""
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, x):
        """Compresses the input data x using the model"""
        x = np.array(x)
        indices = [5, 6, 11, 12, 13]
        x = np.array(x)
        
        ## Subtract the inital value of the nucleus value for all three positions
        x[:, :, 11:14]=x[:, :, 11:14]-x[:, 0, 2][:, np.newaxis, np.newaxis]

        #Now select the relevant indices
        x = x[:, 2*30:, indices]

        #Convert to tensor and necessary device
        x = torch.from_numpy(x).float().to(self.device)
        x = x.flatten(start_dim=1)

        return x


    def __repr__(self):
        return f'Compressor(device={self.device})'