from omegaconf import OmegaConf

from .resnet import Resnet
from .basic_encoder import BasicEncoder
from .dilated_resnet import ResNetV2StraightV2
from DeWrapper.utils import get_pylogger
logger = get_pylogger()


def builder(cfg):
    kwargs = {}
    if cfg.get("kwargs") is not None:
        kwargs = OmegaConf.to_object(cfg.kwargs)

    if cfg.type[0:6] == "resnet":
        num_layers = int(cfg.type[6:])
        backbone = Resnet(num_layers, load_pretrained=False)
    elif cfg.type == "basic_encoder":
        backbone = BasicEncoder(**kwargs)
    elif cfg.type == "dilated_resnet":
        backbone = ResNetV2StraightV2(**kwargs)
    
    return backbone