import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, bias=True,omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        
        # self.in_features = in_feature
            
        # Set trainable parameters if they are to be simultaneously optimized
        # self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        # self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        # self.linear = nn.Linear(in_features,
        #                         out_features,
        #                         bias=bias,
        #                         dtype=dtype)
    
    def forward(self, input):
        # lin = self.linear(input)
        omega = self.omega_0 * input
        scale = self.scale_0 * input
        
        return torch.exp(1j*omega - scale.abs().square()).real