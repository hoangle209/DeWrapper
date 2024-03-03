from torch.autograd import Function, Variable
from torch import nn
import torch

class test(nn.Module):
    def __init__(self):
        super().__init__()
        a = torch.Tensor([1.])
        self.register_buffer('a', a)

    def forward(self, x):
        b = Variable(self.a)
        print(b)

t = test()
x= torch.Tensor([1])
t(x)
for param in t.parameters():
    print(param)