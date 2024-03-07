import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Timer
from torchmetrics import MeanMetric
from kornia.geometry.transform import remap, warp_image_tps

import numpy as np

from DeWrapper.models import STN
from DeWrapper.models.utils.fourier_converter import FourierConverter
from DeWrapper.models.utils.thin_plate_spline import KorniaTPS

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

        self.fourier_converter = FourierConverter(self.cfg)
        self.kornia_tps = KorniaTPS(self.cfg)

        # self.tps = TPS(self.cfg)
        # if self.cfg.loss.mutual.enable:
        #     self.kornia_tps = KorniaTPS(self.cfg)

        # Losses
        self.configure_crit()
 

    # def forward_(self, x):
    #     """Forward function. Using for inference 
        
    #     Parameters:
    #     -----------
    #         x: Tensor, (b, 3, h, w)
    #             normalized input image
    #     """
    #     coarse_mesh = self.coarse_transformer(x)
    #     _, mapX_coarse_, mapY_coarse_ = self.tps(coarse_mesh)
    #     x_coarse = remap(x, 
    #                      mapX_coarse_, mapY_coarse_,
    #                      mode="bilinear", 
    #                      padding_mode="zeros", 
    #                      align_corners=True,
    #                      normalized_coordinates=False) # whether the input coordinates are normalized in the range of [-1, 1].

    #     refine_mesh = self.refine_transformer(x_coarse)
    #     _, mapX_refine_, mapY_refine_ = self.tps(refine_mesh)
    #     x_refine = remap(x_coarse, 
    #                      mapX_refine_, mapY_refine_,
    #                      mode="bilinear", 
    #                      padding_mode="zeros", 
    #                      align_corners=True,
    #                      normalized_coordinates=False)

    #     x_fft_converted = self.fourier_converter(x_refine) # Denoising for friendly OCR

    #     output = {
    #         "x_converted": x_fft_converted, 
    #         "x_refine"   : x_refine,
    #         "coarse_map" : (mapX_coarse_, mapY_coarse_),
    #         "refine_map" : (mapX_refine_, mapY_refine_)
    #     }
    #     return output
    

    def forward(self, x):
        """Forward function. Using for inference 
        
        Parameters:
        -----------
            x: Tensor, (b, 3, h, w)
                normalized input image
        """
        coarse_mesh = self.coarse_transformer(x)
        coarse_kernel_weight_, coarse_affine_weights_ = self.kornia_tps(coarse_mesh)
        x_coarse = warp_image_tps(x, 
                                  coarse_mesh,
                                  coarse_kernel_weight_,
                                  coarse_affine_weights_)
        
        refine_mesh = self.refine_transformer(x_coarse)
        refine_kernel_weight_, refine_affine_weights_ = self.kornia_tps(refine_mesh)
        x_refine = warp_image_tps(x_coarse, 
                                  refine_mesh,
                                  refine_kernel_weight_,
                                  refine_affine_weights_)
        
        x_fft_converted = self.fourier_converter(x_refine) # Denoising for friendly OCR
        
        output = {
            "x_converted": x_fft_converted, 
            "x_refine"   : x_refine,
            "coarse_mesh": coarse_mesh,
            "coarse_map" : (coarse_kernel_weight_, coarse_affine_weights_),
            "refine_mesh": refine_mesh,
            "refine_map" : (refine_kernel_weight_, refine_affine_weights_),
        }
        return output


    def configure_crit(self):
        loss_type = {
            "L1"       : F.l1_loss,
            "L2"       : F.mse_loss,
            "smooth_L1": F.smooth_l1_loss
        }
        
        # Coarse loss
        crit_coarse = self.cfg.trainer.loss.coarse
        if crit_coarse in loss_type:
            self.crit_coarse = loss_type[crit_coarse]
        else:
            logger.warning(f"Coarse loss type {crit_coarse} is not implemented.\
                            Using L1 loss as default...")
            self.crit_coarse = loss_type["L1"]

        # Refinement loss
        crit_refine = self.cfg.trainer.loss.coarse
        if crit_refine in loss_type:
            self.crit_refine = loss_type[crit_refine]
        else:
            logger.warning(f"Refinement loss type {crit_refine} is not implemented.\
                            Using L1 loss as default...")
            self.crit_refine = loss_type["L1"]

        # Mutual loss
        if self.cfg.trainer.loss.mutual.enable:
            """Document with two different geometric distortions can be 
            mutually transformed to each other if their mesh grids are predicted correctly.
            """
            crit_mutual = self.cfg.trainer.loss.mutual.type
            if crit_mutual in loss_type:
                self.crit_mutual = loss_type[crit_mutual]
            else:
                logger.warning(f"Refinement loss type {crit_mutual} is not implemented.\
                                Using L1 loss as default...")
                self.crit_mutual = loss_type["L1"]


    def on_train_start(self):
        torch.cuda.empty_cache()
    

    def step(self, batch):
        img = batch["img"]
        ref = batch["ref"] # ground-truth

        # Fourier Converter
        x_ = self.fourier_converter(img)
        ref_ = self.fourier_converter(ref, is_normalized=False)

        out = self.forward(img)
        coarse_mesh = out["coarse_mesh"]
        coarse_kernel_weight_, coarse_affine_weights_ = out["coarse_map"]
        refine_mesh = out["refine_mesh"]
        refine_kernel_weight_, refine_affine_weights_= out["refine_map"] 

        x_fft_coarse_ = warp_image_tps(x_, 
                                       coarse_mesh,
                                       coarse_kernel_weight_,
                                       coarse_affine_weights_)
        x_fft_refine_ = warp_image_tps(x_fft_coarse_, 
                                       refine_mesh,
                                       refine_kernel_weight_,
                                       refine_affine_weights_)
        loss_coarse = self.crit(x_fft_coarse_, ref_)
        loss_refine = self.crit(x_fft_refine_, ref_)

        if self.cfg.trainer.loss.mutual.enable:
            D1 = batch["deform1"]
            D2 = batch["deform2"]

            d1_mesh = self.coarse_transformer(D1)
            d2_mesh = self.coarse_transformer(D2)
            kernel_weight_1, affine_weights_1 = self.kornia_tps(d1_mesh, d2_mesh) # d1 to d2
            kernel_weight_2, affine_weights_2 = self.kornia_tps(d2_mesh, d1_mesh) # d2 to d1

            wrapedD2 = warp_image_tps(D1, d1_mesh, kernel_weight_1, affine_weights_1)
            wrapedD1 = warp_image_tps(D2, d2_mesh, kernel_weight_2, affine_weights_2)

            loss_mutual = self.crit_mutual(wrapedD1, D1) + self.crit_mutual(wrapedD2, D2)
            w = self.self.cfg.trainer.loss.mutual.weight
            loss_coarse += w * loss_mutual
        
        loss_ = loss_coarse + loss_refine

        return {
            "coarse": loss_coarse,
            "refine": loss_refine,
            "total"  : loss_
        }


    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.train_loss(loss["total"].item())
        for key in loss.keys():
            self.log("train/loss/" + key, loss[key].item(), on_step=False, on_epoch=True, prog_bar=True)
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