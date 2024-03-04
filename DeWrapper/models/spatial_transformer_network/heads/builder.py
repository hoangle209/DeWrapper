from .FDRhead import FDRHead
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

def builder(cfg):
    name = cfg.head.type
    logger.info(f"Building head type {name}")
    if name == "FDRHead":
        head = FDRHead(cfg)
    
    return head