import torch

def compute_weighted_rmse(delta, lat):
    weights = torch.cos(lat)
    weights = weights / weights. mean()
    error = torch.sum((delta)**2, dim=-1, keepdim=True)
    error = weights * error
    loss = error. mean()
    loss = torch.sqrt(loss)
    
    return loss

def compute_weighted_mse(delta, lat):
    weights = torch.cos(lat)
    weights = weights / weights.mean()
    error = torch.sum((delta)**2, dim=-1, keepdim=True)
    error = weights * error
    loss = error. mean()
    
    return loss

def compute_weighted_mae(delta, lat):
    weights = torch.cos(lat)
    weights = weights / weights.mean()
    error = torch.sum(torch.abs(delta), dim=-1, keepdim=True)
    error = weights * error
    loss = error. mean()
    
    return loss

def compute_quantile_ae(delta, lat, quantile=0.99999):
    weights = torch.cos(lat)
    weights = weights / weights.mean()
    loss = torch.quantile(torch.abs(delta), quantile)
    
    return loss

def compute_max_ae(delta, lat):
    weights = torch.cos(lat)
    weights = weights / weights.mean()
    loss = torch.max(torch.abs(delta))
    
    return loss