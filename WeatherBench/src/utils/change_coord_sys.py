import torch

def to_spherical(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    
    theta = torch.arccos(z)  
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], -1)

def to_cartesian(theta, phi):
    # theta, phi = points[..., 2], points[..., 3]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)