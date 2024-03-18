from .FDRhead import FDRHead
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

def builder(cfg):
    if cfg.head.type == "FDRHead":
        head = FDRHead(cfg)
    
    return head