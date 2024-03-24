from omegaconf import OmegaConf

from .fdr_head import FDRHead
from .fiducial_head import FiducialHead
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

def builder(cfg):
    kwargs = {}
    if cfg.get("kwargs") is not None:
        kwargs = OmegaConf.to_object(cfg.kwargs)

    if cfg.type == "fdr":
        head = FDRHead(**kwargs)
    elif cfg.type == "fiducial":
        head = FiducialHead(**kwargs)
    
    return head