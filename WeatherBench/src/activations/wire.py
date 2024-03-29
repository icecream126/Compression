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
    
    def __init__(self, is_first=True, input_dim=5, width = 512, bias=True,omega=10.0, sigma=10.0,
                 trainable=False):
        super().__init__()
        self.is_first = is_first
        self.omega_0 = omega
        self.scale_0 = sigma
        if self.is_first==True:
            self.input_dim = input_dim
        else:
            self.input_dim = width
        self.out_dim = width
        
        # self.in_features = in_feature
            
        # Set trainable parameters if they are to be simultaneously optimized
        # self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        # self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        # self.freqs = nn.Linear(self.input_dim, self.out_dim)
        # self.scale = nn.Linear(self.input_dim, self.out_dim)
        
        # self.linear = nn.Linear(in_features,
        #                         out_features,
        #                         bias=bias,
        #                         dtype=dtype)
    
    def forward(self, input):
        # lin = self.linear(input)
        omega = self.omega_0 * input
        scale = self.scale_0 * input
        
        freq_term = torch.cos(omega)
        gauss_term = torch.exp(-(scale**2))
        
        return freq_term * gauss_term