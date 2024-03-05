import torch

import itertools
import numpy as np

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPS:
    """Thin-plate Spline
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        w = cfg.target_width
        h = cfg.target_height
        grid_w = cfg.grid_width 
        grid_h = cfg.grid_height

        x_ = torch.arange(0, w + 1e-5, w / (grid_w - 1)) * 2 / (w - 1) - 1
        y_ = torch.arange(0, h + 1e-5, h / (grid_h - 1)) * 2 / (h - 1) - 1
        Y, X = torch.meshgrid(y_, x_)
        target_control_points = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1) # (N, 2)
        # target_control_points.float().requires_grad = True # to float32 to set require_grad=True

        N = target_control_points.size(0)
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points) # (N, N)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

        # compute inverse matrix
        self.inverse_kernel = torch.inverse(forward_kernel)
        self.inverse_kernel.float().requires_grad = True
        
        HW = w * h
        target_coordinate = list(itertools.product(range(h), range(w)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        self.target_coordinate_origin = torch.cat([X, Y], dim=1).cpu().numpy() # storing origin target coordinate to use in get_remap function

        # convert coordinate to range(-1, 1)
        Y = Y * 2 / (h - 1) - 1
        X = X * 2 / (w - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        self.target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)
        self.target_coordinate_repr.requires_grad = True

        self.padding_matrix = torch.zeros(1, 3, 2)
        self.padding_matrix.requires_grad = True
    
    
    def get_remap_(self, source_coordinate):
        B, _, c = source_coordinate.size()
        w = self.cfg.target_width
        h = self.cfg.target_height
        source_coord_ = source_coordinate.clone().detach().cpu().numpy()

        source_coord_ = (source_coord_ + 1) * np.array([w, h]) / 2 # mapping to origin coordinate
        index_ = (source_coord_[:, :, 0] + w * source_coord_[:, :, 1]).astype(np.int32)

        mapX = mapY = np.zeros(shape=(B, h*w), dtype=np.float32)
        for i in range(B):
            mapX[i, index_[i]] = self.target_coordinate_origin[..., 0].reshape(-1)
            mapY[i, index_[i]] = self.target_coordinate_origin[..., 1].reshape(-1)

        return mapX.reshape(B, h, w), mapY.reshape(B, h, w)


    def __call__(self, source_control_points):
        B, num_points, _ = source_control_points.size()
        Y = torch.cat([source_control_points, self.padding_matrix.expand(B, 3, 2)], dim=1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        mapX_, mapY_ = self.get_remap_(source_coordinate)
        
        return source_coordinate, mapX_, mapY_