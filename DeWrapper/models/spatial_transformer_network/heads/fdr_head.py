from torch import nn
import torch

from ..blocks.conv import Conv

class FDRHead(nn.Module):
    """Fourier Document Restoration Network Head
    This head is implemented from: https://arxiv.org/abs/2203.09910
    """
    def __init__(self,
                 num_mid_dilate_cv=3,
                 num_end_dilate_cv=2,
                 mid_dim=128, 
                 mid_dim_2=256,
                 en_dim=512,
                 in_channel=None,
                 grid_size=[9, 9]
                       ):
        super().__init__()

        grid_width, grid_height = grid_size
        self.num_mid_dilate_cv = 3
        self.num_end_dilate_cv = 2

        for i in range(self.num_mid_dilate_cv):
            setattr(
                self, f"dilated_cv{i+1}", 
                Conv(in_channel, mid_dim, d=(i+1))
                )
            
        self.cv1 = Conv(mid_dim * 3, mid_dim_2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(self.num_end_dilate_cv):
            setattr(
                self, f"dilated_cv{i+4}", 
                Conv(mid_dim_2, mid_dim_2, d=(i+1))
                )
            
        self.cv2 = Conv(mid_dim_2 * 2, en_dim)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(en_dim, 
                                grid_width * grid_height * 2)
    
    def forward(self, x):
        B, C, H, W = x.size()

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
        out = out.view(B, -1, 2)
        return out        


