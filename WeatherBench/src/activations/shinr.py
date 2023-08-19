import torch
from torch import nn
from math import ceil

from .relu import ReLULayer
import sys
sys.path.append('/home/khm/NNCompression/WeatherBench/src')
from utils.spherical_harmonics import get_spherical_harmonics
from utils.change_coord_sys import to_spherical

class SphericalHarmonicsLayer(nn.Module):
    def __init__(
            self, 
            max_order, 
            omega=10,
            **kwargs,
        ):
        super().__init__()
        self.max_order = max_order
        self.hidden_dim = (max_order+1)**2
        self.omega = omega

        self.linear_t = nn.Linear(1, self.hidden_dim)
        self.linear_p = nn.Linear(1, self.hidden_dim)
        with torch.no_grad():
            self.linear_t.weight.uniform_(-1, 1)
            self.linear_p.weight.uniform_(-1, 1)
        
    def forward(self, input):
        points = to_spherical(input[...,2:])
        theta, phi = points[...,0], points[...,1 ]
        


        sh_list = []
        for l in range(self.max_order+1):
            sh = get_spherical_harmonics(l, phi, theta)
            sh_list.append(sh)

        out = torch.cat(sh_list, dim=-1).squeeze(-2)

        time = input[..., 0:1]
        pressure = input[..., 1:2]
        lin_t = self.linear_t(time)
        lin_p = self.linear_p(pressure)
        
        omega_t = self.omega * lin_t
        omega_p = self.omega * lin_p

        out = out * torch.sin(omega_t) * torch.sin(omega_p)
        return out

class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers,
            max_order,
            time,
            skip=True,
            omega=10.,
            sigma=10.,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(max_order, time, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(self.nonlin((max_order+1)**2,
                                            hidden_dim,
                                            is_first=False,
                                            omega=omega,
                                            sigma=sigma))     
            else:
                if skip and i == ceil(hidden_layers/2):
                    self.net.append(self.nonlin(hidden_dim+input_dim,
                                                hidden_dim,
                                                is_first=False,
                                                omega=omega,
                                                sigma=sigma))
                else:
                    self.net.append(self.nonlin(hidden_dim,
                                                hidden_dim,
                                                is_first=False,
                                                omega=omega,
                                                sigma=sigma))

        final_linear = nn.Linear(hidden_dim, output_dim)
        
        self.net.append(final_linear)
    
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers/2)+1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x