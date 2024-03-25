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
                 grid_size=[9, 9],
                 strides=32,
                 im_size=(768, 1088)
                ):
        super().__init__()

        self.num_mid_dilate_cv = num_mid_dilate_cv
        self.num_end_dilate_cv = num_end_dilate_cv

        for i in range(num_mid_dilate_cv):
            setattr(
                self, f"dilated_cv{i + 1}", 
                Conv(in_channel, mid_dim, d=(i+1))
                )
            
        self.cv1 = Conv(mid_dim * num_mid_dilate_cv, mid_dim_2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(num_end_dilate_cv):
            setattr(
                self, f"dilated_cv{i + 1 + num_mid_dilate_cv}", 
                Conv(mid_dim_2, mid_dim_2, d=(i+1))
                )
            
        self.cv2 = Conv(mid_dim_2 * num_end_dilate_cv, en_dim)
        self.drop = nn.Dropout(p=0.2, inplace=True)

        w, h = im_size[0] // strides // 2, im_size[1] // strides // 2
        grid_width, grid_height = grid_size
        self.linear = nn.Linear(en_dim * w * h, 
                                grid_width * grid_height * 2)
    
    def forward(self, x):
        B, C, H, W = x.size()

        dilate = []
        for i in range(self.num_mid_dilate_cv):
            y_ = getattr(self, f"dilated_cv{i+1}")(x)
            dilate.append(y_)
        dilate = torch.cat(dilate, dim=1)

        x_ = self.cv1(self.pool1(dilate))

        dilate_ = []
        for j in range(self.num_end_dilate_cv):
            y_ = getattr(self, f"dilated_cv{j+1+self.num_mid_dilate_cv}")(x_)
            dilate_.append(y_)
        dilate_ = torch.cat(dilate_, dim=1)

        out = self.linear(self.drop(self.cv2(dilate_).view(B, -1)))
        out = out.view(B, -1, 2)
        return out        


