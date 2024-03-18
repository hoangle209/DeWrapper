from omegaconf import OmegaConf

from .resnet import resnet_builder
from .basic_encoder import BasicEncoder

from DeWrapper.utils import get_pylogger
logger = get_pylogger()


def builder(cfg):
    if cfg.backbone.type[0:6] == "resnet":
        backbone = resnet_builder(cfg.backbone.type)
    elif cfg.backbone.type == "basic_encoder":
        kwargs = OmegaConf.to_object(cfg.backbone.kwargs)
        backbone = BasicEncoder(**kwargs)
    
    return backbone