import torch
from torch import nn
from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import xarray as xr
from tqdm import trange
# from numba import cuda
# from WeatherBench.src.utils.change_coord_sys import to_spherical


def to_spherical(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    
    theta = torch.arccos(z)  
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], -1)

# def plt_wavelet_separate(input, values, out_dim):
#     # Euclidean map / separately
#     spherical = to_spherical(input[...,2:])
#     lat = spherical[...,0].squeeze(0).flatten()
#     lon = spherical[...,1].squeeze(0).flatten()
#     lat = lat.detach().cpu().numpy()
#     lon = lon.detach().cpu().numpy()
    
    
#     np_values = values.detach().cpu().numpy()
#     np_values = np_values.squeeze(0)
    
#     for i in range(out_dim):
#         value = np_values[...,i].flatten()
#         plt.rcParams['font.size'] = 50
#         fig = plt.figure(figsize=(40, 20))
#         plt.tricontourf(
#             lon,
#             -lat,
#             value,
#             levels=100,
#             cmap = 'hot',
#         )
#         plt.colorbar()
#         plt.savefig('fig'+str(i)+'.png', dpi=300)
#         plt.show()
        
        
def plt_wavelet_combined(input, values, current_epoch, mode, run_id):
        
    # Euclidean map / together
    spherical = to_spherical(input[...,2:])
    lat = spherical[...,0].squeeze(0).flatten()
    lon = spherical[...,1].squeeze(0).flatten()
    lat = lat.detach().cpu().numpy()
    lon = lon.detach().cpu().numpy()
    
    
    np_values = values.detach().cpu().numpy()
    if np_values.shape[0] == 1:
        np_values = np_values.squeeze(0)
    
    summed_values = np.sum(np_values,axis=-1)
    min_value = np.min(summed_values)
    max_value = np.max(summed_values)
    normalized_values = (summed_values - min_value) / (max_value - min_value)

    if len(normalized_values.shape)>1:
        normalized_values = normalized_values.flatten()
    plt.rcParams['font.size'] = 50
    fig = plt.figure(figsize=(40, 20))
    plt.tricontourf(
        lon,
        -lat,
        normalized_values,
        levels=100,
        cmap = 'hot',
    )
    plt.colorbar()
    
    figure_dir = './wavelet_figure/'+str(run_id)+'/'
    figure_name = str(mode)+'_'+str(current_epoch)+'.png'
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir+figure_name, dpi=300)
    plt.close()        


# def plt_wavelet_combined(input, values, current_epoch, mode, run_id):
#     batch_size = input.shape[0]
    
#     for batch_idx in range(batch_size):
        
#         curr_input = input[batch_idx]
#         curr_values = values[batch_idx]
#         # Euclidean map / together
#         spherical = to_spherical(curr_input[...,2:])
#         lat = spherical[...,0].squeeze(0).flatten()
#         lon = spherical[...,1].squeeze(0).flatten()
#         lat = lat.detach().cpu().numpy()
#         lon = lon.detach().cpu().numpy()
        
        
#         np_values = curr_values.detach().cpu().numpy()
#         if np_values.shape[0] == 1:
#             np_values = np_values.squeeze(0)
        
#         summed_values = np.sum(np_values,axis=-1)
#         min_value = np.min(summed_values)
#         max_value = np.max(summed_values)
#         normalized_values = (summed_values - min_value) / (max_value - min_value)

#         if len(normalized_values.shape)>1:
#             normalized_values = normalized_values.flatten()
#         plt.rcParams['font.size'] = 50
#         fig = plt.figure(figsize=(40, 20))
#         plt.tricontourf(
#             lon,
#             -lat,
#             normalized_values,
#             levels=100,
#             cmap = 'hot',
#         )
#         plt.colorbar()
        
#         figure_dir = './wavelet_figure/'+str(run_id)+'/'+str(batch_idx)+'/'
#         figure_name = str(mode)+'_'+str(current_epoch)+'.png'
#         if not os.path.isdir(figure_dir):
#             os.makedirs(figure_dir)
#         plt.savefig(figure_dir+figure_name, dpi=300)
#         plt.close()
                
                
# def plt_error_map(input, pred, target, current_epoch, mode, run_id):
    
#     batch_size = input.shape[0]
#     # input.shape : [3, 43320, 5]
#     # curr_input.shape : [43320, 5]
#     # pred, target shape : [3, 43320, 1]
#     # curr_pred, curr_target shape : [43320, 1]

#     for batch_idx in range(batch_size):
            
#         curr_input = input[batch_idx]
#         curr_pred = pred[batch_idx]
#         curr_target = target[batch_idx]
#         # Euclidean map / together
#         spherical = to_spherical(curr_input[...,2:])
#         lat = spherical[...,0].squeeze(0).flatten()
#         lon = spherical[...,1].squeeze(0).flatten()
        
#         weights = torch.cos(lat)
#         weights = weights / weights.mean()
#         error = torch.sum((curr_pred - curr_target)**2 * weights ,dim=-1,keepdim=True)
        
#         curr_target_min, curr_target_max = curr_target.min(), curr_target.max()
#         curr_target = (curr_target - curr_target_min) / (curr_target_max - curr_target_min)
#         curr_pred = (curr_pred - curr_target_min) / (curr_target_max - curr_target_min)
#         curr_pred = torch.clip(curr_pred, 0, 1)
        
        
#         if len(curr_target.shape)>1:
#             curr_target = curr_target.flatten()
#         if len(curr_pred.shape)>1:
#             curr_pred = curr_pred.flatten()
#         if len(error.shape)>1:
#             error = error.flatten()
        
        
#         lat = lat.detach().cpu().numpy()
#         lon = lon.detach().cpu().numpy()
#         error = error.detach().cpu().numpy()
        
#         plt.rcParams['font.size'] = 50
#         fig = plt.figure(figsize=(40, 20))
#         plt.tricontourf(
#             lon,
#             -lat,
#             error,
#             levels=100,
#             cmap='hot',
#         )
#         plt.colorbar()
        
#         figure_dir = './error_map/'+str(wandb.run.id)+'/'+str(batch_idx)+'/'
#         figure_name = str(mode)+'_'+str(current_epoch)+'.png'
#         if not os.path.isdir(figure_dir):
#             os.makedirs(figure_dir)
#         plt.savefig(figure_dir+figure_name, dpi=300)
#         plt.close()
        
def plt_error_map(model, current_epoch = None, logger=None, logger_id=None):
    model.eval()
    data_path='./'
    file_name='dataset1.nc'
    ds = xr.open_dataset(f"{data_path}/{file_name}")
    # ds_pred = xr.zeros_like(ds['z']) - 9999 # (361, 720, 1)
    ds = ds.assign_coords(time=ds.time.dt.dayofyear-1)
    dtype = torch.float32
    lat = torch.tensor(np.array(ds.latitude), dtype=torch.float32, device="cuda") # min, max : -90, 90
    lon = torch.tensor(np.array(ds.longitude), dtype=torch.float32, device="cuda")
    ps = np.array(ds.level).astype(float) # (11,) min: 10.0 / max : 1000.0
    # ts = np.array(ds.time).astype(float) # (366, ) min : 0.0 / max : 365.0
    for j in trange(ps.shape[0]):
        t = torch.tensor([0], device="cuda", dtype=torch.float32)
        p = torch.tensor([float(ps[j])], device="cuda",dtype=torch.float32)
        # draw error map only for 10, 500, 1000 for efficiency
        if int(p)==10 or int(p)==500 or int(p)==1000:
            coord = torch.stack(torch.meshgrid(t, p, lat, lon, indexing="ij"), dim=-1).squeeze(0).squeeze(0) # check if (361, 720, 4)
            target = ds['z'][0,j,:,:]
            
            pred = model(coord)
            pred = pred.detach().cpu().numpy().squeeze(-1)
            
            lat_vec = coord[:,:,2]
            lon_vec = coord[:,:,3]
            weights = torch.cos(lat_vec)
            weights = weights / weights.mean()
            
            
            if len(pred.shape)>2:
                pred = pred.squeeze(-1)
            if len(target.shape)>2:
                target = target.squeeze(-1)
            
            
            delta = np.array(pred - target)
            
            
            weights =weights.detach().cpu().numpy()
            error = (delta)**2 * weights
            
            
            lat_vec = lat_vec.flatten()
            lon_vec = lon_vec.flatten()
            error = error.flatten()
            
            
            lat_vec = lat_vec.detach().cpu().numpy()
            lon_vec = lon_vec.detach().cpu().numpy()
            # error = error.detach().cpu().numpy()
            
            
            plt.rcParams['font.size'] = 50
            fig = plt.figure(figsize=(40, 20))
            plt.tricontourf(
                lon_vec,
                -lat_vec,
                error,
                levels=100,
                cmap='hot',
            )
            
            plt.colorbar()
            
            figure_dir = './error_map/'+str(logger_id)+'/train/'
            figure_name = str(current_epoch)+'_0_'+str(int(p))+'.png'
            if not os.path.isdir(figure_dir):
                os.makedirs(figure_dir)
            plt.savefig(figure_dir+figure_name, dpi=300)
            plt.close()
        else:
            continue
    model.train()
        
def plt_error_map_test(lat_vec, lon_vec, delta, p, logger):
    weights = torch.cos(lat_vec)
    weights = weights / weights.mean()
    
    # weights =weights.detach().cpu().numpy()
    
    error = (delta)**2 * weights
    
    lat_vec = lat_vec.flatten()
    lon_vec = lon_vec.flatten()
    error = error.flatten()
    
    lat_vec = lat_vec.detach().cpu().numpy()
    lon_vec = lon_vec.detach().cpu().numpy()
    error = error.detach().cpu().numpy()
    
    plt.rcParams['font.size'] = 50
    fig = plt.figure(figsize=(40, 20))
    plt.tricontourf(
        lon_vec,
        -lat_vec,
        error,
        levels=100,
        cmap='hot',
    )
    plt.colorbar()
    
    figure_dir = './error_map/'+str(logger.experiment.id)+'/test/'
    figure_name = '0_'+str(p)+'.png'
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    plt.savefig(figure_dir+figure_name, dpi=300)
    plt.close()