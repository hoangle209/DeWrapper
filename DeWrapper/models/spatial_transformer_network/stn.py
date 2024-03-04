from torch import nn

from .backbones.builder import builder as backbone_builder
from .heads.builder import builder as head_builder
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class STN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.backbone = backbone_builder(self.cfg)
        self.cfg.model.backbone.out_channel = self.backbone.channels[-1]
        self.head = head_builder(self.cfg)
    
    def forward(self, x):
        x = self.head(self.backbone(x))
        return x


        