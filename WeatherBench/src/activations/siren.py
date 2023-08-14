import torch
import numpy as np
from torch import nn
from math import ceil
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega.
    
        If is_first=True, omega is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(
            self, 
            omega=10.0,
            **kwargs,
        ):
        super().__init__()

        self.omega = omega
        # self.is_first = is_first
        
        # self.input_dim = input_dim
        # self.linear = nn.Linear(input_dim, output_dim)
        
        # with torch.no_grad():
        #     if self.is_first:
        #         self.linear.weight.uniform_(-1 / self.input_dim, 
        #                                      1 / self.input_dim)      
        #     else:
        #         self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega, 
        #                                      np.sqrt(6 / self.input_dim) / self.omega)
        
    def forward(self, input):
        return torch.sin(self.omega * input)