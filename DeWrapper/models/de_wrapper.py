import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Timer
from torchmetrics import MeanMetric
from kornia.geometry.transform import remap

import numpy as np

from DeWrapper.models import STN
from DeWrapper.models.utils.fourier_converter import FourierConverter
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
        """
        Parameters:
        -----------
            cfg, DictConfig
                omegaconf instance
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()

        self.best_val_acc = 0

        # Models
        logger.info("Creating Coarse Transformer...")
        self.coarse_transformer = STN(self.cfg)
        logger.info("Creating Refine Transformer...")
        self.refine_transformer = STN(self.cfg)

        self.FFT = FourierConverter(self.cfg)
        self.TPS = TPS(self.cfg)

        # Losses
        self.configure_crit()
 
    def forward(self, x):
        """Forward function. Using for inference 
        
        Parameters:
        -----------
            x: Tensor, (b, 3, h, w)
                normalized input image
        """
        coarse_mesh = self.coarse_transformer(x)
        _, mapX_coarse_, mapY_coarse_ = self.TPS(coarse_mesh)
        x_coarse = remap(x, 
                         mapX_coarse_, mapY_coarse_,
                         mode="bilinear", 
                         padding_mode="zeros", 
                         align_corners=True,
                         normalized_coordinates=False) # whether the input coordinates are normalized in the range of [-1, 1].

        refine_mesh = self.refine_transformer(x_coarse)
        _, mapX_refine_, mapY_refine_ = self.TPS(refine_mesh)
        x_refine = remap(x_coarse, 
                         mapX_refine_, mapY_refine_,
                         mode="bilinear", 
                         padding_mode="zeros", 
                         align_corners=True,
                         normalized_coordinates=False)

        x_fft_converted = self.FFT(x_refine) # Denoising for friendly OCR
        return x_fft_converted, x_refine

    def configure_crit(self):
        loss_type = self.cfg.train.loss_type
        
        if loss_type == "L1":
            self.crit = F.l1_loss
        elif loss_type == "smooth_L1":
            self.crit = F.smooth_l1_loss
        elif loss_type == "L2":
            self.crit = F.mse_loss
        else:
            logger.warning(f"{loss_type} is not implemented. 
                           Using L1 loss as default...")
            self.crit = F.l1_loss

    def on_train_start(self):
        torch.cuda.empty_cache()
    
    def step(self, batch):
        img = batch["img"]
        ref = batch["ref"] # ground-truth
        x_fft_converted, _ = self.forward(img)
        ref_ = self.FFT(ref)

        loss = self.crit(x_fft_converted, ref_)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.train_loss(loss.item())
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_iter_stats(batch_idx)

        del batch
        return loss

    def log_iter_stats(self, cur_iter): # TODO
        def gpu_mem_usage():
            """Computes the GPU memory usage for the current device (MB)."""
            mem_usage_bytes = torch.cuda.max_memory_allocated()
            return mem_usage_bytes / 1024 / 1024
        
        if(cur_iter%self.cfg.log_frequency != 0):
            return 0
        
        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches),
                "train_loss": "%.4f"%(self.train_loss.compute().item()),
                "val_loss": "%.4f"%(self.val_loss.compute().item()),
                "time": "%.4f"%(self.timer.time_elapsed()-self.timer_last_iter),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
                "mem": int(np.ceil(mem_usage)),
            }
            self.timer_last_iter = self.timer.time_elapsed()
        except:
            self.timer = Timer()
            self.timer_last_iter = self.timer.time_elapsed()
            stats = {}
            
        self.train_loss.reset()
        self.val_loss.reset()
        
        logger.info(stats)