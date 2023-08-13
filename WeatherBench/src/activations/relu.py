import torch
from torch import nn
from math import ceil
    
class ReLULayer(nn.Module):
    def __init__(
            self, 
            **kwargs,
        ):
        super().__init__()
        # self.input_dim = input_dim
        # self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, input):
        return nn.functional.relu(input)
    
class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers, 
            skip=True,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.nonlin = ReLULayer
            
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(input_dim, hidden_dim))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers/2):
                self.net.append(self.nonlin(hidden_dim+input_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers/2)+1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x