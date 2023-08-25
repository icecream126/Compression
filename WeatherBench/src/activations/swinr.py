import torch
from torch import nn
from math import pi, ceil

from .relu import ReLULayer
from WeatherBench.src.utils.change_coord_sys import to_cartesian

class SphericalGaborLayer(nn.Module):
    def __init__(
            self, 
            input_dim=1,
            width=512,
            omega=0.01,
            sigma=0.1,
            tpdim=2,
            **kwargs,
        ):
        super().__init__()

        self.omega = omega
        self.sigma = sigma
        self.out_dim = width
        self.tpdim = tpdim

        self.dilate = nn.Parameter(torch.empty(1, self.out_dim))
        nn.init.normal_(self.dilate)

        self.u = nn.Parameter(torch.empty(self.out_dim))
        self.v = nn.Parameter(torch.empty(self.out_dim))
        self.w = nn.Parameter(torch.empty(self.out_dim))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)

        self.linear_tp = nn.Linear(self.tpdim, self.out_dim)

    def forward(self, input):
        zeros = torch.zeros(self.out_dim, device=self.u.device)
        ones = torch.ones(self.out_dim, device=self.u.device)

        alpha = 2*pi*self.u
        beta = torch.arccos(torch.clamp(2*self.v-1, -1+1e-6, 1-1e-6))
        gamma = 2*pi*self.w
        
        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)

        Rz_alpha = torch.stack([
            torch.stack([cos_alpha, -sin_alpha, zeros], 1), 
            torch.stack([sin_alpha,  cos_alpha, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        Rx_beta = torch.stack([
            torch.stack([ ones,     zeros,      zeros], 1), 
            torch.stack([zeros, cos_beta, -sin_beta], 1), 
            torch.stack([zeros, sin_beta,  cos_beta], 1)
            ], 1)

        Rz_gamma = torch.stack([
            torch.stack([cos_gamma, -sin_gamma, zeros], 1), 
            torch.stack([sin_gamma,  cos_gamma, zeros], 1), 
            torch.stack([    zeros,      zeros,  ones], 1)
            ], 1)
        
        R = torch.bmm(torch.bmm(Rz_gamma, Rx_beta), Rz_alpha)

        points = input[...,-3:]
        points = torch.matmul(R, points.unsqueeze(-2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        dilate = torch.exp(self.dilate)

        freq_arg = 2 * dilate * x / (1e-6+1+z)
        gauss_arg = 4 * dilate * dilate * (1-z) / (1e-6+1+z)

        time_pressure = input[..., :-3]
        lin_tp = self.linear_tp(time_pressure)
        freq_arg = freq_arg + lin_tp
        gauss_arg = gauss_arg + lin_tp * lin_tp

        freq_term = torch.cos(self.omega*freq_arg)
        gauss_term = torch.exp(-self.sigma*self.sigma*gauss_arg)
        
        return freq_term * gauss_term
    

class INR(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim,
            hidden_dim, 
            hidden_layers,
            time,
            skip=True,
            omega=10.,
            sigma=10.,
            **kwargs,
        ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalGaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(hidden_dim, time, omega, sigma))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
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