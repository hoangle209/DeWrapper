from .FDRhead import FDRHead
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

def builder(cfg):
    logger.info(f"      - Head <{cfg.head.type}>")
    if cfg.head.type == "FDRHead":
        head = FDRHead(cfg)
    
    return head