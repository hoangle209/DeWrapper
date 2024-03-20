import torch
import torch.nn as nn
from kornia.geometry.transform import get_tps_transform

class KorniaTPS(nn.Module):
    """Using built-in of kornia function
    """
    def __init__(self, target, grid_size=(9,9)):
        super().__init__()

        doc_w, doc_h = target
        grid_w, grid_h = grid_size 
        self.x = torch.linspace(0, doc_w - 1, steps=grid_w) / doc_w * 2 - 1
        self.y = torch.linspace(0, doc_h - 1, steps=grid_h) / doc_h * 2 - 1

        Y, X = torch.meshgrid(self.y, self.x, indexing='ij')
        self.target_control_points = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1) # (N, 2)
    
    def forward(self, control_points, target_points=None):
        if target_points is None:
            B, N, _ = control_points.size()
            target_points = self.target_control_points.expand(B, N, 2).to(control_points.device)

        kernel_weight, affine_weights = get_tps_transform(target_points, control_points)
        return kernel_weight, affine_weights