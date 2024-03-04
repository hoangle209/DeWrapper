import torch

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

        x_ = torch.arange(0, w + 1e-5, w / (grid_w - 1))
        y_ = torch.arange(0, h + 1e-5, h / (grid_h - 1))
        Y, X = torch.meshgrid(y_, x_)
        target_control_points = torch.cat([Y.reshape(-1, 1), X.reshape(-1, 1)], dim=1) # (81, 2)
        target_control_points.float().requires_grad = True # to float32 to set require_grad=True

        N = target_control_points.size(0)
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        inverse_kernel = inverse_kernel.float().require_grad = True
        