from .resnet import resnet_builder
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

def builder(cfg):
    name = cfg.backbone.type
    logger.info(f"Building backbone type {name}")
    if "resnet" in name:
        backone = resnet_builder(name)
    
    return backone