import torch
from torch import nn
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torchmetrics import MeanMetric

import cv2

from DeWrapper.models import STN
from DeWrapper.models.utils.fourier_converter import FFT
from DeWrapper.models.utils.thin_plate_spline import TPS
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
        self.FFT = FFT(self.cfg)
        self.TPS = TPS(self.cfg)

        
    def forward(self, x):
        """
        """
        X_origin = x["origin_img"]
        x = x["img"]
        coarse_mesh = self.coarse_transformer(x)
        _, mapX_coarse_, mapY_coarse_ = self.TPS(coarse_mesh)
        x_coarse = cv2.remap(X_origin, 
                             mapX_coarse_[0], mapY_coarse_[0], # TODO: handle mapping in batch
                             cv2.INTER_AREA, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=0)

        refine_mesh = self.refine_transformer(x)
        _, mapX_refine_, mapY_refine_ = self.TPS(refine_mesh)
        x_refine = cv2.remap(x_coarse, 
                             mapX_refine_[0], mapY_refine_[0], # TODO: handle mapping in batch
                             cv2.INTER_AREA, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=0)

        x_ = self.FFT.converter(x_refine)

        return x_, x_refine


    def loss(self):
        pass

    def on_train_start(self):
        torch.cuda.empty_cache()
        