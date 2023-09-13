import torch

def compute_weighted_rmse(delta, lat):
    min_lat = torch.min(lat)
    if min_lat < -2:
        lat_rad = torch.deg2rad(lat)
        weights = torch.cos(lat_rad) # lat의 범위 한번 확인해보고 그 다음에 확인
    else:
        weights = torch.cos(lat)
    
    weights /= weights.mean() # check weights mean and sum of training dataset
    
    loss = torch.sqrt(((delta)**2 * weights).mean())
    
    return loss

def compute_weighted_mse(delta, lat):
    min_lat = torch.min(lat)
    if min_lat <-2:
        weights = torch.cos(torch.deg2rad(lat)) # lat의 범위 한번 확인해보고 그 다음에 확인
    else:
        weights = torch.cos(lat)
    weights = torch.cos(torch.deg2rad(lat))
    weights /= weights.mean()
    loss = ((delta)**2 * weights).mean()
    
    return loss

def compute_weighted_mae(delta, lat):
    min_lat = torch.min(lat)
    if min_lat <-2:
        weights = torch.cos(torch.deg2rad(lat)) # lat의 범위 한번 확인해보고 그 다음에 확인
    else:
        weights = torch.cos(lat)
    weights /= weights.mean()
    loss = (torch.abs(delta)*weights).mean()
    
    return loss

def compute_quantile_ae(delta, lat, quantile=0.99999):
    loss = torch.quantile(torch.abs(delta), quantile)
    
    return loss

def compute_max_ae(delta, lat):
    loss = torch.max(torch.abs(delta))
    
    return loss