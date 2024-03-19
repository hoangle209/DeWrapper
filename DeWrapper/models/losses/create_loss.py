import torch.nn as nn
from omegaconf import OmegaConf

from .ssim import SSIM, MS_SSIM
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class Loss(nn.Module):
    loss_type = {
            "L1"       : nn.L1Loss,
            "L2"       : nn.MSELoss,
            "smooth_L1": nn.SmoothL1Loss,
            "ssim"     : SSIM,
            "ms-ssim"  : MS_SSIM
        }
    
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.type in Loss.loss_type:
            kwargs = {}
            if self.cfg.loss.coarse.get("kwargs") is not None:
                kwargs = OmegaConf.to_object(cfg.kwargs)
            
            self.crit = Loss.loss_type[cfg.type](**kwargs)
        else:
            logger.warning(f"Loss type {cfg.type} is not implemented.\
                            Using L1 loss as default !!!")
            self.crit = Loss.loss_type["L1"]()
            self.cfg.type = "L1"
    
    def forward(self, X, Y):
        B, C, H, W = X.size()

        loss = self.crit(X, Y)
        if self.cfg.type in ["L1", "L2", "smooth_L1"]:
            loss = loss.mean(dim=(1,2,3))
            loss = loss.mean()
        elif self.cfg.type in ["ssim", "ms-ssim"]:
            loss = loss.mean(dim=1)
            loss = 100 * (1 - loss.mean())

        return loss.float()
        

