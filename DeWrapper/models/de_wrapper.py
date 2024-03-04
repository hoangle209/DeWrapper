from torch import nn
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torchmetrics import MeanMetric

from DeWrapper.models import STN
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class DeWrapper(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """
    def __init__(self, cfg):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()

        self.best_val_acc = 0

        self.coarse_transformer = STN(self.cfg)
        self.refine_transformer = STN(self.cfg)
        
    
    def forward(self, x):
        pass