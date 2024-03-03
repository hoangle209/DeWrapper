from torch import nn

from .backbones.builder import builder as backbone_builder
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class STN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = backbone_builder(cfg.model)

        