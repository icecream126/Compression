import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import get_model_size_mb
import math
import numpy as np
import xarray as xr
from argparse import ArgumentParser
from types import SimpleNamespace
from datetime import datetime
import matplotlib.pyplot as plt
import pyinterp
import pyinterp.backends.xarray
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange, tqdm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from WeatherBench.src.activations.swinr import SphericalGaborLayer
from WeatherBench.src.activations.wire import ComplexGaborLayer
from WeatherBench.src.activations.siren import SineLayer
from WeatherBench.src.activations.shinr import SphericalHarmonicsLayer
from WeatherBench.src.utils.loss import compute_weighted_mae, compute_weighted_rmse, compute_weighted_mse, compute_quantile_ae, compute_max_ae
from WeatherBench.src.utils.visualize import plt_error_map, plt_error_map_test
import sys, os
import wandb



YEAR = 2016

activation_dict = {
    'relu': nn.ReLU,
    'siren': SineLayer,
    'wire': ComplexGaborLayer,
    'shinr': SphericalHarmonicsLayer,
    'swinr': SphericalGaborLayer,
    'gelu': nn.GELU
}

class ERA5stat():
    def __init__(self, file_name_mean, file_name_std, data_path, variable, grid_type):
        self.ds_mean = xr.load_dataset(f"{data_path}/{file_name_mean}")[variable]
        self.ds_std = xr.load_dataset(f"{data_path}/{file_name_std}")[variable]
        self.grid_type = grid_type
        if grid_type == "regular":
            self.interp_mean = pyinterp.backends.xarray.Grid3D(self.ds_mean)
            self.interp_std = pyinterp.backends.xarray.Grid3D(self.ds_std)
        elif grid_type == "sphere_grid":
            self.interp_mean = RegularGridInterpolator((self.ds_mean.plev, self.ds_mean.y, self.ds_mean.x), self.ds_mean.data)
            self.interp_std = RegularGridInterpolator((self.ds_std.plev, self.ds_std.y, self.ds_std.x), self.ds_std.data)
    
    def interp_regular(self, plev, lat, lon):
        mean = self.interp_mean.trivariate(
            dict(longitude=lon.ravel(),
                latitude=lat.ravel(),
                level=plev.ravel())).reshape(lat.shape)
        std = self.interp_std.trivariate(
            dict(longitude=lon.ravel(),
                latitude=lat.ravel(),
                level=plev.ravel())).reshape(lat.shape)
        return mean, std

    def interp_sphere_grid(self, plev, y, x):
        coord = torch.stack((plev, lat, lon))
        mean = self.interp_mean(coord).reshape(y.shape)
        std = self.interp_mean(coord).reshape(y.shape)
        return mean, std

class WeatherBenchDataset_sampling(Dataset):
    def __init__(self, file_name, data_path, nbatch, nsample, variable="z"):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_mfdataset(file_path).load()
        self.ds = self.ds.assign_coords(time=np.arange(len(self.ds.time)))
        #self.grid = pyinterp.Grid3D(pyinterp.Axis(self.ds.time), pyinterp.Axis(self.ds.lat), pyinterp.Axis(self.ds.lon, is_circle=True), self.ds[variable].data)
        self.interpolator = RegularGridInterpolator((self.ds.time, self.ds.lat, self.ds.lon), self.ds[variable].data, bounds_error=False, fill_value=None)
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        self.mean = np.array(self.ds[variable].mean(dim=["time"]))
        self.std = (self.ds[variable].max(dim=["time"]) - np.array(self.ds[variable].min(dim=["time"])))
        self.interp_mean = RegularGridInterpolator((self.ds.lat, self.ds.lon), self.mean, bounds_error=False, fill_value=None)
        self.interp_std = RegularGridInterpolator((self.ds.lat, self.ds.lon), self.std, bounds_error=False, fill_value=None)
    
    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rnds = self.rndeng.draw(self.nsample)
            time = rnds[:, 0] * (self.ntime - 1)
            pind = torch.zeros_like(time) + float(self.ds.level.mean())
            latind = 90 - 180/math.pi*torch.acos(1 - 2 * rnds[:, 1])
            lonind = (rnds[:, 2] * 360)
            coord = torch.stack((time, pind, latind, lonind), dim=-1).to(torch.float32)
            #var_sampled = pyinterp.trivariate(self.grid, time.ravel(), latind.ravel(), lonind.ravel()).reshape(latind.shape)
            coord_in = torch.stack((time, latind, lonind), dim=-1)
            var_sampled = self.interpolator(coord_in).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1).to(torch.float32)
            mean = torch.as_tensor(self.interp_mean(coord_in[..., 1:]).reshape(var_sampled.shape)).to(torch.float32)
            std = torch.as_tensor(self.interp_std(coord_in[..., 1:]).reshape(var_sampled.shape)).to(torch.float32)
            return coord, var_sampled, mean, std

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(np.array(self.ds.lat))
        lon_v = torch.as_tensor(np.array(self.ds.lon))
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + float(self.ds.level.mean())
        t = torch.zeros_like(lat) + float(tind)
        coord = torch.stack((t, p, lat, lon), dim=-1).unsqueeze(0).to(torch.float32)
        var = torch.as_tensor(np.array(self.ds[self.variable].isel(time=tind))).unsqueeze(-1).unsqueeze(0).to(torch.float32)
        mean = torch.as_tensor(self.mean).reshape(var.shape).to(torch.float32)
        std = torch.as_tensor(self.std).reshape(var.shape).to(torch.float32)
        return coord, var, mean, std
        
class ERA5Dataset_sampling(Dataset):
    # dataset = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
    def __init__(self, file_name, data_path, nbatch, nsample, variable="z", stat_config=None):
        file_path = f"{data_path}/{file_name}"
        self.ds = xr.open_dataset(file_path)[variable].load()#{"time": 20}
        self.ds = self.ds.assign_coords(time=self.ds.time.dt.dayofyear-1) # (366, 11, 361, 720)
        # this inputs raw xarray and this outputs axis organized GRID 4D (x : longitude / y : latitude / z : time / u : pressure)
        self.interpolator = pyinterp.backends.xarray.Grid4D(self.ds) 
        self.variable = variable
        self.ntime = len(self.ds.time)
        self.nbatch = nbatch
        self.nsample = nsample
        self.rndeng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        # self.mean_lat_weight = torch.cos(self.rndeng.draw(self.nsample)[:,1]).mean()
        if stat_config is not None:
            self.stat = ERA5stat(**stat_config)
        else:
            self.stat = None
        #assert len(sample_block_size) == 3 # np, nlat, nlon

    def __len__(self):
        return self.nbatch # 24093

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # rnds[:,0] min, max : tensor(8.8131e-06) tensor(1.0000)
            # rnds[:,1] min, max : tensor(1.0652e-05) tensor(1.0000)
            # rnds[:,2] min, max : tensor(2.0703e-06) tensor(1.0000)
            # time      min, max : tensor(0.0032) tensor(364.9950)
            # pind      min, max : tensor(10.) tensor(1000.)
            # latind    min, max : tensor(-89.5058) tensor(89.6262)
            # lonind    min, max : tensor(0.0007) tensor(359.9936)
            
            rnds = self.rndeng.draw(self.nsample) # self.nsample : 43320, rnds : torch.Size(43320, 3)
            time = rnds[:, 0] * (self.ntime - 1) # Sobolev sequence에 time을 곱한다는 것은 sampling을 하는 느낌
            #pind = (torch.rand((self.nsample,)) * (1000-10) + 10)
            # self.ds.level : <xarray.DataArray 'level' (level:11)>
            # array([ 10, 50, 100, 200, 300, 400, 500, 700, 850, 925, 1000])
            # torch.randperm(self.nsample) % len(self.ds.level) (sample 수만큼 Randperm하면 최대 43320까지의 범위내에서 randperm이 되는데 이거를 level의 범위로 나눠줌으로써 0부터 10까지의 범위로 randmperm됨 )
            # 각 sample에 대한 pind를 갖고 오는 것 같음
            pind = torch.as_tensor(np.array(self.ds.level), dtype=torch.float32)[torch.randperm(self.nsample) % len(self.ds.level)]
            # latind = (torch.rand((self.nsample,)) * 180 - 90)
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            # mean_lat_weight = self.mean_lat_weight
            latind = 90 - 180/math.pi*torch.acos(1 - 2 * rnds[:, 1])
            lonind = (rnds[:, 2] * 360)
            coord = torch.stack((time, pind, latind, lonind), dim=-1) # torch.Size([43320,4])
            var_sampled = self.interpolator.quadrivariate(
                dict(longitude=lonind.ravel(),
                     latitude=latind.ravel(),
                     time=time.ravel(),
                     level=pind.ravel())).reshape(latind.shape)
            var_sampled = torch.as_tensor(var_sampled).unsqueeze(-1) #torch.Size([43320,1])
            if self.stat is None:
                return coord, var_sampled # torch.Size([43320,4]), torch.Size([43320,1])
            else:
                mean, std = self.stat.interp_regular(pind, latind, lonind)
                return coord, var_sampled, mean, std# , mean_lat_weight

    def getslice(self, tind, pind):
        lat_v = torch.as_tensor(np.array(self.ds.latitude)) # 361
        lon_v = torch.as_tensor(np.array(self.ds.longitude)) # 720
        lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
        p = torch.zeros_like(lat) + np.array(self.ds.level)[pind] # torch.zeros_like (lat) : shape (361,720)의 tensor에다가 pind의 어떤 pressure 값을 전부 더해줌 (즉, (361, 720) dimension의 pressure 값 생성)
        t = torch.zeros_like(lat) + float(tind) 
        coord = torch.stack((t, p, lat, lon), dim=-1).to(torch.float32) # (361, 720, 4)
        # self.ds.isel(time=tind, level=pind) # (361, 720) (전체 (366, 11, 361, 720) dim의 data에서 해당하는 time, pressure의 지점을 뽑아온듯)
        var = torch.as_tensor(np.array(self.ds.isel(time=tind, level=pind))).to(torch.float32).unsqueeze(-1) # (361, 720, 1)
        # mean_lat_weight = self.mean_lat_weight
        return coord.unsqueeze(0), var.unsqueeze(0)# coord.unsqueeze(0) : (1, 361, 720, 4) / var.unsqueeze(0) : (1, 361, 720, 1)

class FourierFeature(nn.Module):
    def __init__(self, sigma, infeature, outfeature):
        super(FourierFeature, self).__init__()
        self.feature_map = nn.Parameter(torch.normal(0., sigma, (outfeature, infeature)) ,requires_grad=False)
    def forward(self, x, cos_only: bool = False):
        # x shape: (..., infeature)
        x = 2*math.pi*F.linear(x, self.feature_map)
        if cos_only:
            return torch.cos(x)
        else:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
    
class LonLat2XYZ(nn.Module):
    def forward(self, x):
        # x shape: torch.Size([1,361, 720, 4])
        # time, p, lat, lon .shape : torch.Size([1,361, 720])
        time = x[..., 0]
        p = x[..., 1]
        lat = x[..., 2] # min(-1.5708), max(1.5708)
        lon = x[..., 3] # min(0), max(6.2745)
        sinlat = torch.sin(lat) # min(-1), max(1)
        coslat = torch.cos(lat) # min(-4.3711e-08), max(1)
        sinlon = torch.sin(lon) # min(-1), max(1)
        coslon = torch.cos(lon) # min(-1), max(1)
        return torch.stack((time, p, sinlat, coslat*sinlon, coslat*coslon), dim=-1)
    
class NormalizeInput(nn.Module):
    def __init__(self, tscale, zscale):
        super(NormalizeInput, self).__init__()
        self.scale = nn.Parameter(torch.tensor([1.0/tscale, 1.0/zscale, math.pi/180., math.pi/180.]), requires_grad=False)
    def forward(self, x):
        return x*self.scale
    
class InvScale(nn.Module):
    def forward(self, coord, z_normalized):
        factor = 0.9
        p = coord[..., 1:2]
        std = 0.385e5-0.35e4*torch.log(p)
        mean = 4.315e5-6.15e4*torch.log(p)
        return (z_normalized / factor)*std + mean

# class ResBlock(nn.Module):
#     def __init__(self, width, activation, use_batchnorm=True, use_skipconnect=True, shinr_layer_index=None, input_dim=None, depth=None, first_activation=None):
#         super(ResBlock, self).__init__()
#         self.width = width
#         self.wavelet_dim = width
#         self.fc1 = nn.Linear(width, width, bias=False)
#         self.fc2 = nn.Linear(width, width, bias=True)
#         self._fc1 = nn.Linear(self.wavelet_dim, width, bias=False)
#         self.use_batchnorm = use_batchnorm
#         self.use_skipconnect = use_skipconnect
#         self.activation = activation
#         self.depth = depth
#         self.first_activation=first_activation
#         if use_batchnorm:
#             self._bn1 = nn.BatchNorm1d(self.wavelet_dim)
#             self.bn1 = nn.BatchNorm1d(width)
#             self.bn2 = nn.BatchNorm1d(width)

#     def forward(self, x_original):
#         # x shape: (batch_size, width)
#         if self.depth==0:
#             x = x_original
#             if self.use_batchnorm:
#                 if self.first_activation=='swinr':
#                     x = self._bn1(x)
#                 else:
#                     x = self.bn1(x)
#             x = self.activation(x)
#             # x = F.gelu(x)
#             if self.first_activation == 'swinr':
#                 x = self._fc1(x)
#             else:
#                 x = self.fc1(x)
#             if self.use_batchnorm:
#                 x = self.bn2(x)
#             x = self.activation(x)
#             # x = F.gelu(x)
#             x = self.fc2(x)

#             return x
#         else:
#             x = x_original
#             if self.use_batchnorm:
#                 x = self.bn1(x)
#             x = self.activation(x)
#             # x = F.gelu(x)
#             x = self.fc1(x)
#             if self.use_batchnorm:
#                 x = self.bn2(x)
#             x = self.activation(x)
#             # x = F.gelu(x)
#             x = self.fc2(x)
#             if self.use_skipconnect:
#                 return x + x_original
#             else:
#                 return x
class ResBlock(nn.Module):
    def __init__(self, width, activation, use_batchnorm=True, use_skipconnect=True, shinr_layer_index=None, input_dim=None):
        super(ResBlock, self).__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width, bias=False)
        self.fc2 = nn.Linear(width, width, bias=True)
        self.use_batchnorm = use_batchnorm
        self.use_skipconnect = use_skipconnect
        self.activation = activation
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(width)
            self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x_original):
        # x shape: (batch_size, width)
        x = x_original
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.activation(x)
        # x = F.gelu(x)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.activation(x)
        # x = F.gelu(x)
        x = self.fc2(x)
        if self.use_skipconnect:
            return x + x_original
        else:
            return x

class MultiResolutionEmbedding(nn.Module):
    def __init__(self, feature_size, nfeature, tresolution, tscale):
        # tresolution: timestep size in hours
        super().__init__()
        self.tscale = tscale
        self.tresolution = tresolution
        self.embed1 = nn.Embedding(366, nfeature, max_norm=1.0)
        self.embed2 = nn.Embedding(24, nfeature, max_norm=1.0)
        self.embed3 = nn.Embedding(int(feature_size/tscale), nfeature, max_norm=1.0)

    def forward(self, idx):
        idx = idx.squeeze(-1)
        idx1 = torch.floor(idx * self.tresolution).long()
        idx2 = torch.floor(idx / self.tscale).long()
        embed1 = self.embed1((idx1 // 24) % 366)
        embed2 = self.embed2(idx1 % 24)
        embed3 = self.embed3(idx2)
        embed = torch.cat((embed1, embed2, embed3), dim=-1)
        return embed
    
class FitNet(nn.Module):
    __constants__ = ['use_xyztransform','use_fourierfeature','use_tembedding','use_invscale', 'depth']

    def __init__(self, args):
        super(FitNet, self).__init__()
        self.args = args
        if args.use_invscale:
            self.invscale = InvScale()
        if args.use_xyztransform:
            self.lonlat2xyz = LonLat2XYZ()
            ns = 3
        else:
            ns = 2
        if args.use_tembedding:
            ne = args.ntfeature * 3
            self.embed_t = MultiResolutionEmbedding(args.tembed_size, args.ntfeature, args.tresolution, args.tscale)
        else:
            ne = 0
        if args.use_fourierfeature:
            self.fourierfeature_t = FourierFeature(args.f_sigma, 1, args.ntfeature)
            self.fourierfeature_p = FourierFeature(args.f_sigma, 1, args.nfeature)
            if args.first_activation=='shinr' or args.first_activation == 'swinr':
                nf = 2*(args.nfeature+args.ntfeature)+ns            
            else : # swinr and shinr should preserve its input shape
                self.fourierfeature_s = FourierFeature(args.f_sigma, ns, args.nfeature)
                nf = 2*(2*args.nfeature + args.ntfeature) # 바깥쪽 2*는 sin, cos 해주는 경우인 것 같음 # 안쪽 2*는 p랑 s에서 모두 nf만큼 크기를 가지니까 저렇게
            
        else:
            nf = 2 + ns
        self.normalize = NormalizeInput(args.tscale, args.zscale)     
        self.depth = args.depth
        
        # First activation
        if args.first_activation=='swinr':
            if args.use_fourierfeature:
                self.first_activation = activation_dict[args.first_activation](tpdim = nf - ns, width = args.wavelet_dim, omega=args.omega, sigma = args.sigma, mode = args.swinr_mode)
            else:
                self.first_activation = activation_dict[args.first_activation](width = args.wavelet_dim, omega=args.omega, sigma = args.sigma, mode = args.swinr_mode)
        elif args.first_activation=='wire':
            self.first_activation = activation_dict[args.first_activation](width = args.width, is_first=True, omega = args.omega, sigma = args.sigma)
        elif args.first_activation=='siren':
            self.first_activation = activation_dict[args.first_activation](omega0 = args.omega)
        elif args.first_activation=='shinr':
            if args.use_fourierfeature:
                self.first_activation = activation_dict[args.first_activation](tpdim = nf-ns, max_order = args.max_order)
            else:
                self.first_activation = activation_dict[args.first_activation](max_order = args.max_order)
            args.width = (args.max_order+1)**2
        else:
            self.first_activation = activation_dict[args.first_activation]()
        
        # Next activation
        if args.activation=='wire': 
            self.activation = activation_dict[args.activation](is_first=False, omega = args.omega, sigma= args.sigma)
        else:
            self.activation = activation_dict[args.activation]()
            
        
        self.fci = nn.Linear(nf + ne, args.width)
        #     def __init__(self, width, activation, use_batchnorm=True, use_skipconnect=True, shinr_layer_index=None, input_dim=None, depth=None, first_activation=None):
        # self.fcs = nn.ModuleList([ResBlock(width = args.width, activation = self.activation, use_batchnorm = args.use_batchnorm, use_skipconnect = args.use_skipconnect, depth = i, first_activation = args.first_activation) for i in range(args.depth)])
        self.fcs = nn.ModuleList([ResBlock(width = args.width, activation = self.activation, use_batchnorm = args.use_batchnorm, use_skipconnect = args.use_skipconnect) for i in range(args.depth)])
        
        self.fco = nn.Linear(args.width, 1)

        if args.first_activation == 'shinr':
            self.use_xyztransform = False
        else:
            self.use_xyztransform = True
        
        self.use_fourierfeature = self.args.use_fourierfeature
        self.use_tembedding = self.args.use_tembedding 
        self.use_invscale = self.args.use_invscale

    def forward(self, coord):
        # coord.shape : torch.Size([1, 361, 720, 4])
        batch_size = coord.shape[:-1] # 4
        x = self.normalize(coord) # torch.Size([1, 361, 720, 4])
        if self.use_xyztransform:
            x = self.lonlat2xyz(x)
        if self.use_fourierfeature:
            t = x[..., 0:1] # torch.Size([1, 361, 720, 1])
            p = x[..., 1:2] # torch.Size([1, 361, 720, 1])
            s = x[..., 2:] # torch.Size([1, 361, 720, 1])
            
            if self.args.first_activation == 'swinr' and self.args.first_activation == 'shinr':
                x = torch.cat((self.fourierfeature_t(t), self.fourierfeature_p(p), s), dim=-1) # torch.Size([1,361,720,291])
            else:
                x = torch.cat((self.fourierfeature_t(t), self.fourierfeature_p(p), self.fourierfeature_s(s)), dim=-1) # torch.Size([1, 361, 720, 544])
             
        if self.use_tembedding:
            x = torch.cat((self.embed_t(coord[..., 0:1]), x), dim=-1)
        

        if args.first_activation=='siren' or args.first_activation=='relu' or args.first_activation=='gelu' or args.first_activation=='wire':
            x = self.fci(x) # torch.Size([1,361,720,5 / 544]) -> torch.Size([1, 361, 720, 64])
        # print('before x : ',x.shape)
        # print('coord lat shape : ',x[...,2:3].shape)
        # print('lat : ',x[...,2:3])
        # print('lon : ',x[...,3:])
        
        x = self.first_activation(x)
        # print('after x : ',x.shape)
        # x = F.gelu(self.fci(x))
        x = x.flatten(end_dim=-2) # batchnorm 1d only accepts (N, C) shape
        for fc in self.fcs:        
            x = fc(x)
        # print('before x 2: ',x.shape)
        x = self.activation(x)
        # print('after x 2: ',x.shape)
        # x = F.gelu(x)
        x = self.fco(x)
        x = x.view(batch_size).unsqueeze(-1)
        if self.use_invscale or self.args.use_stat:
            x = torch.tanh(x)
        if self.use_invscale:
            x = self.invscale(coord, x)
        return x
    
class FitNetModule(pl.LightningModule):
    # sigma=1.5, omega=30., nfeature=256, width=512, depth=4, tscale=60.0, zscale=100., learning_rate=1e-3, batch_size=3
    def __init__(self, args, logger):
        super(FitNetModule, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = FitNet(args)#torch.jit.script(FitNet(args))
        self.input_type = torch.float32
        self.last_training_pred = None
        self.last_valid_pred = None
        
        self.last_training_input = None
        self.last_valid_input = None
        
        self.last_training_target = None
        self.last_valid_target = None
        
        self.mylogger = logger
        
    def train_dataloader(self):
        if self.args.dataloader_mode == "sampling_nc":
            dataset = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
        elif self.args.dataloader_mode == "weatherbench":
            dataset = WeatherBenchDataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
        # dataset[0][0],dataset[1][0] : torch.Size([43320, 4])
        # dataset[0][1],dataset[1][1] : torch.Size([43320, 1])
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, prefetch_factor=8)
        return dataloader # len(dataloader) : 8031
    
    def val_dataloader(self):
        # TODO: use xarray unstack?
        it = 42
        ip = 4
        if self.args.dataloader_mode == "sampling_nc":
            data = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable).getslice(it, ip)
        elif self.args.dataloader_mode == "weatherbench":
            data = WeatherBenchDataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable).getslice(it, ip)
        dataset = TensorDataset(*data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
    
       
    def test_dataloader(self):
        if self.args.dataloader_mode == "sampling_nc":
            dataset = ERA5Dataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
        elif self.args.dataloader_mode == "weatherbench":
            dataset = WeatherBenchDataset_sampling(self.args.file_name, self.args.data_path, 2677*9, 361*120, variable=self.args.variable)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, prefetch_factor=8)
        return dataloader
        
    def configure_optimizers(self):
        # print('wandb args learning_rate : ',args.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10000)
        sched = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'train_weighted_mse'
        }
        return [optimizer], [sched]
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # check if var shape is torch.Size([361, 720]) since delta of test is torch.Size([361, 720])
        if self.args.use_stat:
            coord, var, mean, std = batch
            #print(coord.mean(), var.mean(), mean.mean(), std.mean())
            var_pred = self(coord) * 0.5 * 1.4 * std + mean
        else:
            coord, var = batch
            var_pred = self(coord)
        self.last_training_pred = var_pred
        self.last_training_input = coord
        self.last_training_pred = var_pred
        self.last_training_target = var
        lat = coord[..., 2:3] / 180. * math.pi
        p = coord[..., 1:2]
        assert var.shape == var_pred.shape
        assert var.shape == lat.shape
        delta = var_pred - var # check the shape of delta shape if it is  [361, 720]
        delta_abs = torch.abs(delta)
        loss_linf = delta_abs.max()
        loss_l1 = delta_abs.mean()
        loss_l2 = delta.pow(2).mean()
        if self.args.loss_type == "scaled_mse":
            loss = (delta/(11 - torch.log(p))).pow(2).mean()
        elif self.args.loss_type == "mse":
            loss = loss_l2
        elif self.args.loss_type == "logsumexp":
            loss = torch.logsumexp(torch.abs(delta))
        elif self.args.loss_type == 'weighted_mse':
            loss = compute_weighted_mse(delta, lat)

        # self.log("train_weighted_mse", loss)
        #self.log("train_loss"+self.args.loss_type, loss)
        self.log("train_loss_l2", loss_l2)
        self.log("train_loss_l1", loss_l1)
        self.log("train_loss_linf", loss_linf)
        # self.log("train_weighted_rmse", compute_weighted_rmse(var_pred, var, lat, mean_lat_weight))
        self.log("train_weighted_rmse", compute_weighted_rmse(delta, lat))
        self.log("train_weighted_mse", compute_weighted_mse(delta, lat))
        self.log("train_weighted_mae", compute_weighted_mae(delta, lat))
        self.log("train_quantile_ae", compute_quantile_ae(delta, lat))
        self.log("train_max_ae", compute_max_ae(delta, lat))
        # self.log("weighted_acc", compute_weighted_acc(var_pred, var))
        return loss

    def test_step(self, batch, batch_idx):
        # return self.training_step(batch, batch_idx)
        if self.args.use_stat:
            coord, var, mean, std = batch
            #print(coord.mean(), var.mean(), mean.mean(), std.mean())
            var_pred = self(coord) * 0.5 * 1.4 * std + mean
        else:
            coord, var = batch
            var_pred = self(coord)
        lat = coord[..., 2:3] / 180. * math.pi
        p = coord[..., 1:2]
        assert var.shape == var_pred.shape
        assert var.shape == lat.shape
        delta = var_pred - var
        delta_abs = torch.abs(delta)
        loss_linf = delta_abs.max()
        loss_l1 = delta_abs.mean()
        loss_l2 = delta.pow(2).mean()
        if self.args.loss_type == "scaled_mse":
            loss = (delta/(11 - torch.log(p))).pow(2).mean()
        elif self.args.loss_type == "mse":
            loss = loss_l2
        elif self.args.loss_type == "logsumexp":
            loss = torch.logsumexp(torch.abs(delta))
        elif self.args.loss_type == 'weighted_mse':
            loss = compute_weighted_mse(delta, lat)

        # self.log("train_weighted_mse", loss)
        #self.log("train_loss"+self.args.loss_type, loss)
        self.log("test_loss_l2", loss_l2)
        self.log("test_loss_l1", loss_l1)
        self.log("test_loss_linf", loss_linf)
        # self.log("train_weighted_rmse", compute_weighted_rmse(var_pred, var, lat, mean_lat_weight))
        self.log("test_weighted_rmse", compute_weighted_rmse(delta, lat))
        self.log("test_weighted_mse", compute_weighted_mse(delta, lat))
        self.log("test_weighted_mae", compute_weighted_mae(delta, lat))
        self.log("test_quantile_ae", compute_quantile_ae(delta, lat))
        self.log("test_max_ae", compute_max_ae(delta, lat))
        # self.log("weighted_acc", compute_weighted_acc(var_pred, var))
        return loss
    
    def validation_step(self, batch, batch_idx):
        # batch[0] : torch.Size([1,361,720,4])
        # batch[1] : torch.Size([1,361,720,1])
        
        with torch.no_grad():
            if self.args.use_stat:
                coord, var, mean, std = batch
                var_pred = self(coord) * 0.5 * 1.4 * std + mean
            else:
                coord, var = batch
                var_pred = self(coord)
                
            self.last_valid_pred = var_pred
            self.last_valid_input = coord
            lat = coord[..., 2:3] / 180. * math.pi
            assert var.shape == var_pred.shape
            assert var.shape == lat.shape
            delta_origin = var_pred - var
            delta = delta_origin
            val_loss = delta.abs().max()
            val_loss_l2 = delta.pow(2).mean()
            plt.figure(figsize=(10,8))
            X, Y = coord[..., 3].squeeze().detach().cpu(), coord[..., 2].squeeze().detach().cpu()
            plt.contour(X, Y, var.squeeze().detach().cpu(), colors="green")
            plt.contour(X, Y, var_pred.squeeze().detach().cpu(), colors="red")#cmap="Reds"
            plt.pcolormesh(X, Y, delta_origin.squeeze().detach().cpu(), cmap="coolwarm", shading='nearest')
            plt.axis('scaled')
            plt.title(f'p={torch.mean(coord[..., 1]).item()}')
            plt.colorbar(fraction=0.02, pad=0.04)
            if self.trainer.is_global_zero:
                self.log("val_loss", val_loss, rank_zero_only=True, sync_dist=True)
                self.log("val_loss_l2", val_loss_l2, rank_zero_only=True, sync_dist=True)
                # self.log("weighted_rmse", compute_weighted_rmse(var_pred, var, lat, mean_lat_weight))
                # self.log("weighted_mae", compute_weighted_mae(var_pred, var, lat, mean_lat_weight))
                # self.log("weighted_mae", compute_weighted_mse(var_pred, var, lat, mean_lat_weight))
                
                self.log("val_weighted_rmse", compute_weighted_rmse(delta, lat), rank_zero_only=True, sync_dist=True)
                self.log("val_weighted_mse", compute_weighted_mse(delta, lat), rank_zero_only=True, sync_dist=True)
                self.log("val_weighted_mae", compute_weighted_mae(delta, lat), rank_zero_only=True, sync_dist=True)
                self.log("val_quantile_ae", compute_quantile_ae(delta, lat), rank_zero_only=True, sync_dist=True)
                self.log("val_max_ae", compute_max_ae(delta, lat), rank_zero_only=True, sync_dist=True)

    def on_train_epoch_end(self):
        my_id = None
        if self.trainer.global_rank==0:
            my_id = self.mylogger.experiment.id
            # print('plotting error map')
            plt_error_map(self.model, self.current_epoch, self.mylogger, self.mylogger.experiment.id)
            # print('finish plotting')
            if self.args.first_activation=='swinr':
                coord = torch.tensor(np.load('./coord.npy'), device='cuda')
                coord = self.model.lonlat2xyz(coord)
                self.model.first_activation(coord, visualize=True, current_epoch = self.current_epoch, mode='train', run_id = self.mylogger.experiment.id)
    
        # if my_id:  
        #     with torch.no_grad():
        #         plt_error_map(self.model, self.mylogger, my_id)
                  
        # # visualize first_activation of swinr
        # with torch.no_grad():
        #     if self.args.first_activation == 'swinr':
        #         input = self.model.lonlat2xyz(self.last_training_input)
        #         # save wavelet visualization
        #         self.model.first_activation(input, visualize=True, current_epoch = self.current_epoch, mode='train', run_id = logger.experiment.name)
        #         # save error map visualization
        #         plt_error_map(input, self.last_training_pred, self.last_training_target, current_epoch = self.current_epoch, mode='train', run_id = wandb.run.id)
            
def test_on_wholedataset(quantized_size, file_name, data_path, output_path, output_file, model, device="cuda", variable="z", logger=None):
    
    # plt_error_map(model, 'test')
    lonlat2xyz = LonLat2XYZ()
    ds = xr.open_dataset(f"{data_path}/{file_name}")
    ds_pred = xr.zeros_like(ds[variable]) - 9999 # (361, 720, 1)
    ds = ds.assign_coords(time=ds.time.dt.dayofyear-1)
    dtype = model.input_type # torch.float32
    lat = torch.tensor(np.array(ds.latitude), dtype=dtype, device=device) # min, max : -90, 90
    lon = torch.tensor(np.array(ds.longitude), dtype=dtype, device=device)
    ps = np.array(ds.level).astype(float) # (11,) min: 10.0 / max : 1000.0
    ts = np.array(ds.time).astype(float) # (366, ) min : 0.0 / max : 365.0
    model = model.to(device)
    max_error = np.zeros(ps.shape[0])
    weighted_rmse = np.zeros((ts.shape[0], ps.shape[0]))
    weighted_mse = np.zeros((ts.shape[0], ps.shape[0]))
    weighted_mae = np.zeros((ts.shape[0], ps.shape[0]))
    quantile_ae = np.zeros((ts.shape[0], ps.shape[0]))
    for i in trange(ts.shape[0]):
        for j in range(ps.shape[0]):
            ti = float(ts[i])
            pj = float(ps[j])
            t = torch.tensor([ti], dtype=dtype, device=device)
            p = torch.tensor([pj], dtype=dtype, device=device)
            coord = torch.stack(torch.meshgrid(t, p, lat, lon, indexing="ij"), dim=-1).squeeze(0).squeeze(0) # check if (361, 720, 4)
            with torch.no_grad():
                var_pred = model(coord) 
                ds_pred.data[i, j, :, :] = var_pred.cpu().numpy().squeeze(-1)
                delta = np.array(ds_pred.data[i, j, :, :] - ds[variable][i, j, :, :])
                max_error[j] = max(max_error[j], np.abs(delta).max())
                lat_vec = coord[:,:,2]
                lon_vec = coord[:,:,3]
                delta = torch.tensor(delta, device=device)
                weighted_rmse[i][j] = compute_weighted_rmse(delta, lat_vec)
                weighted_mse[i][j] = compute_weighted_mse(delta, lat_vec)
                weighted_mae[i][j] = compute_weighted_mae(delta, lat_vec)
                quantile_ae[i][j] = compute_quantile_ae(delta, lat_vec)
                if i==0:
                    plt_error_map_test(torch.deg2rad(lat_vec), torch.deg2rad(lon_vec), delta, int(pj), logger)
                

    avg_weighted_rmse = np.mean(weighted_rmse,axis=0)
    avg_weighted_mse = np.mean(weighted_mse,axis=0)
    avg_weighted_mae = np.mean(weighted_mae,axis=0)
    avg_quantile_ae = np.mean(quantile_ae,axis=0)
    first_activation = str(model.args.first_activation)
    model_size = str(quantized_size)
    run_id = str(logger.experiment.id)
    data_path = '/'+first_activation+'_'+model_size+'_'+run_id

    np.save('./weighted_rmse'+data_path,weighted_rmse)
    np.save('./weighted_mse'+data_path,weighted_mse)
    np.save('./weighted_mae'+data_path,weighted_mae)
    np.save('./quantile_mae'+data_path,quantile_ae)
    np.save('./max_ae'+data_path,np.array(max_error))
    
    logger.experiment.define_metric("test_max_ae", step_metric='custom_step')
    logger.experiment.define_metric("test_weighted_rmse", step_metric='custom_step')
    logger.experiment.define_metric("test_weighted_mae", step_metric='custom_step')
    logger.experiment.define_metric("test_weighted_mse", step_metric='custom_step')
    logger.experiment.define_metric("test_quantile_ae", step_metric='custom_step')
    for j in range(ps.shape[0]):
        log_dict={
            "text_max_ae":max_error[j],
            "test_weighted_rmse":avg_weighted_rmse[j],
            "test_weighted_mae":avg_weighted_mae[j],
            "test_weighted_mse":avg_weighted_mse[j],
            "test_quantile_ae":avg_quantile_ae[j],
        }
        
        logger.experiment.log(log_dict)
        
        
        # logger.experiment.log({'test_max_ae': max_error[j],'pressure': ps[j]})
        # logger.experiment.log({'test_weighted_rmse': avg_weighted_rmse[j],'pressure': ps[j] })
        # logger.experiment.log({'test_weighted_mae': avg_weighted_mae[j],'pressure': ps[j] })
        # logger.experiment.log({'test_weighted_mse': avg_weighted_mse[j],'pressure': ps[j]})
        # logger.experiment.log({'test_quantile_ae': avg_quantile_ae[j],'pressure': ps[j]})
        
        # wandb.log({'test_max_ae': max_error[j],'pressure': ps[j]})
        # wandb.log({'test_weighted_rmse': avg_weighted_rmse[j],'pressure': ps[j] })
        # wandb.log({'test_weighted_mae': avg_weighted_mae[j],'pressure': ps[j] })
        # wandb.log({'test_weighted_mse': avg_weighted_mse[j],'pressure': ps[j]})
        # wandb.log({'test_quantile_ae': avg_quantile_ae[j],'pressure': ps[j]})
        

    ds_pred.to_netcdf(f"{output_path}/{output_file}")

def generate_outputs(model, output_path, output_file, device="cuda"):
    file_name = model.args.file_name
    data_path = model.args.data_path
    variable = model.args.variable #"z"
    ds = xr.open_mfdataset(f"{data_path}/{file_name}").load()
    out_ds = xr.zeros_like(ds)
    mean = np.array(ds[variable].mean(dim=["time"]))
    std = (ds[variable].max(dim=["time"]) - np.array(ds[variable].min(dim=["time"])))
    assert len(ds[variable].shape) == 3
    lon_v = torch.as_tensor(np.array(ds.lon), device=device, dtype=torch.float32)
    lat_v = torch.as_tensor(np.array(ds.lat), device=device, dtype=torch.float32)
    lat, lon = torch.meshgrid((lat_v, lon_v), indexing="ij")
    p = torch.zeros_like(lat, device=device) + float(ds.level.mean())
    t = torch.zeros_like(lat, device=device)
    model = model.to(device)
    errors = np.zeros(len(ds.time))
    for it in tqdm(range(len(ds.time))):
        coord = torch.stack((t + it, p, lat, lon), dim=-1)
        with torch.no_grad():
            var_pred = model(coord).squeeze(-1).cpu().numpy() * 0.5 * 1.4 * std + mean
            out_ds[variable].data[it, :, :] = var_pred[:, :]
            var = np.array(ds[variable].isel(time=it))
            errors[it] = np.abs(var_pred - var).max()
    file_name = f"{output_path}/{output_file}"
    print(f"Saving to {file_name}")
    out_ds.to_netcdf(file_name)
    print(errors.max())

def main(args):
    
    # Training
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_weighted_mse" if args.validation else "train_weighted_mse",
        mode="min",
        filename="best"
    )
    # wandb.finish()

    logger = WandbLogger(
        config=args, 
        project="final_compression",
        name='comp/'+args.first_activation+'/'+args.dataset,
    )
    
    model = FitNetModule(args, logger)


    if args.ckpt_path != "":
        model_loaded = FitNetModule.load_from_checkpoint(args.ckpt_path)
        model.model.load_state_dict(model_loaded.model.state_dict())

    trainer = None
    if not args.notraining:
        strategy = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
        trainer = pl.Trainer(profiler='simple',accumulate_grad_batches=args.accumulate_grad_batches, logger=logger, check_val_every_n_epoch=10, accelerator="gpu", auto_select_gpus=True, devices=args.num_gpu, strategy=strategy, min_epochs=10, max_epochs=args.nepoches, gradient_clip_val=0.5, sync_batchnorm=True)
        # trainer = pl.Trainer(log_every_n_steps=1, callbacks=[lrmonitor_cb, checkpoint_cb], logger = logger, accumulate_grad_batches=args.accumulate_grad_batches, check_val_every_n_epoch=10, accelerator="gpu",  min_epochs=10, max_epochs=args.nepoches, gradient_clip_val=0.5, sync_batchnorm=True)# 혹시 몰라서 해둔건데 사용 안할듯
        # trainer = pl.Trainer(profiler='simple',accumulate_grad_batches=args.accumulate_grad_batches, check_val_every_n_epoch=10, accelerator="gpu", auto_select_gpus=True, devices=args.num_gpu, min_epochs=10, max_epochs=args.nepoches, gradient_clip_val=0.5, sync_batchnorm=True)
        trainer.fit(model)

    model.eval()
    # trainer.test(model)
    if (not trainer) or trainer.is_global_zero:
            model_size=get_model_size_mb(model)
            print("Model size (MB):", model_size)
            # wandb.log({'model_size':model_size})
            logger.experiment.log({'model_size': model_size})

    if args.quantizing:
        model.model.fcs = model.model.fcs.half()
        quantized_size = get_model_size_mb(model)
        model.model.fcs = model.model.fcs.float()
        print(f"Quantized (FP16) size (MB): {quantized_size}")
        # wandb.log({'quantized_size':quantized_size})
        logger.experiment.log({"quantized_size": quantized_size})

    if args.testing and ((not trainer) or trainer.is_global_zero):
        test_on_wholedataset(quantized_size, model.args.file_name, model.args.data_path, model.args.output_path, model.args.output_file, model, variable=model.args.variable, logger=logger)

    if args.generate_full_outputs and ((not trainer) or trainer.is_global_zero):
        generate_outputs(model, args.output_path, args.output_file)

    return model

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wavelet_dim', default=None, type=int)
    parser.add_argument('--swinr_mode', default=None, type=str)
    parser.add_argument('--omega', default=0.1, type=float)
    parser.add_argument('--sigma', default=3.0, type=float)
    parser.add_argument('--max_order', default=3, type=int)
    parser.add_argument('--dataset', default='ERA5', type=str)
    parser.add_argument('--validation', default=True, type=bool)
    parser.add_argument('--first_activation', default='gelu', type=str)
    parser.add_argument('--activation', default='gelu', type=str)
    parser.add_argument("--num_gpu", default=-1, type=int)
    parser.add_argument("--nepoches", default=20, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--f_sigma", default=1.6, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--nfeature", default=128, type=int)
    parser.add_argument("--ntfeature", default=16, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--tscale", default=60., type=float)
    parser.add_argument("--zscale", default=100., type=float)
    parser.add_argument("--variable", default="z", type=str)
    parser.add_argument("--dataloader_mode", default="sampling_nc", type=str)
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--use_skipconnect', action='store_true')
    parser.add_argument('--use_invscale', action='store_true')
    parser.add_argument('--use_fourierfeature', action='store_true', default=False)
    parser.add_argument('--use_tembedding', action='store_true')
    parser.add_argument("--tembed_size", default=400, type=int) # number of time steps
    parser.add_argument("--tresolution", default=24, type=float)
    parser.add_argument('--use_xyztransform', action='store_true')
    parser.add_argument('--use_stat', action='store_true')
    parser.add_argument('--loss_type', default="scaled_mse", type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--generate_full_outputs', action='store_true')
    parser.add_argument("--output_path", default=".", type=str)
    parser.add_argument("--output_file", default="output.nc", type=str)
    parser.add_argument('--notraining', action='store_true')
    parser.add_argument('--quantizing', action='store_true')
    # parser.add_argument('--node_rank', default=0, type=int)
    # parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    
    # wandb.init(config=args)
    if args.all:
        args.use_batchnorm = True
        args.use_invscale = not args.use_stat
        args.use_skipconnect = True
        args.use_xyztransform = True
        if args.first_activation == 'relu' or args.first_activation=='gelu':
            args.use_fourierfeature = True
        else:
            args.use_fourierfeature = False
    # print('args.learning_rate ; ',args.learning_rate)

    main(args)