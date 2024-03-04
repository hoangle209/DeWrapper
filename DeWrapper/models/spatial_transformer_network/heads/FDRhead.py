from torch import nn
import torch

from ..blocks.conv import Conv

class FDRHead(nn.Module):
    """Fourier Document Restoration Network Head
    This head is implemented from: https://arxiv.org/abs/2203.09910
    """
    expansion= 1

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        mid_dim = 512
        mid_dim_2 = 1024
        en_dim = 1280
        self.num_mid_dilate_cv = 3
        self.num_end_dilate_cv = 2

        in_channel = cfg.model.backbone.out_channel
        for i in range(3):
            setattr(self, f"dilated_cv{i+1}", 
                    Conv(in_channel, mid_dim, d=2**i))
        self.cv1 = Conv(mid_dim*3, mid_dim_2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        for j in range(2):
            setattr(self, f"dilated_cv{j+4}", 
                    Conv(mid_dim_2, mid_dim_2, d=2**i))
        self.cv2 = Conv(mid_dim_2*2, en_dim)

        num_points = cfg.model.head.num_grid_w * cfg.model.head.num_grid_h
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(en_dim, num_points)
    
    def forward(self, x):
        dilate = []
        for i in range(3):
            y_ = getattr(self, f"dilated_cv{i+1}")(x)
            dilate.append(y_)
        dilate = torch.cat(dilate, dim=1)

        x_ = self.cv1(self.pool1(dilate))

        dilate_ = []
        for j in range(2):
            y_ = getattr(self, f"dilated_cv{j+4}")(x_)
            dilate_.append(y_)
        dilate_ = torch.cat(dilate_, dim=1)

        out = self.linear(self.drop(self.pool2(self.cv2(dilate_)).flatten(1)))
        return out        


