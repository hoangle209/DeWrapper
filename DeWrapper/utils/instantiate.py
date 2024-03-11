from omegacof import OmegaConf, DictConfig
from typing import List
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, LearningRateMonitor, Timer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from .ema_checkpoints import EMACheckpoint, EMA
from .pylogger import get_pylogger
logger = get_pylogger()

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig !!!")
    
    for cb_key in callbacks_cfg:
        if cb_key == "model_summary":
            keys_ = OmegaConf.to_object(callbacks_cfg.get(cb_key))
            cb_ = RichModelSummary(**keys_)
        elif cb_key == "rich_progress_bar":
            keys_ = OmegaConf.to_object(callbacks_cfg.get(cb_key))
            cb_ = RichProgressBar(**keys_)
        elif cb_key == "learning_rate_monitor":
            cb_ = LearningRateMonitor()
        elif cb_key == "timer":
            cb_ = Timer()
        elif cb_key == "model_checkpoint":
            keys_ = OmegaConf.to_object(callbacks_cfg.get(cb_key))
            cb_ = EMACheckpoint(**keys_)
        elif cb_key == "ema":
            keys_ = OmegaConf.to_object(callbacks_cfg.get(cb_key))
            cb_ = EMA(**keys_)
        
        callbacks.append(cb_)    
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    log: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")
    
    for log_key in logger_cfg:
        if log_key == "tensorboard":
            keys_ = OmegaConf.to_object(logger_cfg.get(log_key))
            log_ = TensorBoardLogger(**keys_)
        log.append(log_)
    return log