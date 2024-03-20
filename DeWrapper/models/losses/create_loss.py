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
            "ms-ssim"  : MS_SSIM,
        }
    
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.type in ["L1", "L2", "smooth_L1"]:
            self.crit = Loss.loss_type[cfg.type](reduction='none')
        elif cfg.type in ["ssim", "ms-ssim"]:
            self.crit = Loss.loss_type[cfg.type](data_range=255.0, size_average=False)
        elif cfg.type == "mix":
            self.crit_L1 = Loss.loss_type["L1"](reduction='none')
            self.crit_ms_ssim = Loss.loss_type["ms-ssim"](data_range=255.0, size_average=False)
        else:
            logger.warning(f"Loss type {cfg.type} is not implemented.\
                            Using L1 loss as default !!!")
            self.crit = Loss.loss_type["L1"](reduction='none')
            self.cfg.type = "L1"
    
    def forward(self, X, Y):
        B, C, H, W = X.size()

        if self.cfg.type == "mix":
            alpha = 0.84
            loss_l1 = self.crit_L1(X, Y).mean(dim=(1,2,3))
            loss_ms_ssim = self.crit_ms_ssim(X, Y)
            loss = (1 - alpha) * loss_l1.mean() + alpha * 255.0 * loss_ms_ssim.mean()
        else:
            loss = self.crit(X, Y)
            if self.cfg.type in ["L1", "L2", "smooth_L1"]:
                loss = loss.mean(dim=(1,2,3))
                loss = loss.mean()
            elif self.cfg.type in ["ssim", "ms-ssim"]:
                loss = loss.mean(dim=1)
                loss = 255.0 * (1 - loss.mean())

        return loss.float()
        

