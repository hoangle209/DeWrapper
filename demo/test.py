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

from DeWrapper.models.spatial_transformer_network.stn import STN
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    model = STN(cfg)
    # print(model)