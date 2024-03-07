import sys
sys.path.insert(1, ".")

# from torch.autograd import Function, Variable
# from torch import nn
# import torch

# class test(nn.Module):
#     def __init__(self):
#         super().__init__()
#         a = torch.Tensor([1.])
#         self.register_buffer('a', a)

#     def forward(self, x):
#         b = Variable(self.a)
#         print(b)

# t = test()
# x= torch.Tensor([1])
# t(x)
# for param in t.parameters():
#     print(param)

from omegaconf import DictConfig, OmegaConf
import cv2
import torch
import numpy as np

if __name__ == "__main__":
    from DeWrapper.models.de_wrapper import DeWrapper
    cfg = OmegaConf.load("config/default.yaml")
    model = DeWrapper(cfg)
    oimg = torch.ones(1, 3, 224, 512)
    gimg = np.ones((224, 512), dtype=np.uint8)
    img = torch.ones(1, 3, 512, 224)
    i = {
        "origin_img": oimg,
        "gray_img": gimg,
        "img": img,        
    }
    model(img)

    # dataloader = {
    #     "img": preprocess(img),
    #     "ref": gt_img
    # }