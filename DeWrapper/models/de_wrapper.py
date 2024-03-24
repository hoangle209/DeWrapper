import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from kornia.geometry.transform import warp_image_tps
from kornia.augmentation import RandomThinPlateSpline
import numpy as np

from DeWrapper.models import STN
from DeWrapper.models.utils.fourier_converter import FourierConverter
from DeWrapper.models.utils.thin_plate_spline import KorniaTPS
from DeWrapper.models.losses.create_loss import Loss

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

random_tps = RandomThinPlateSpline(scale=0.15, p=1.0, keepdim=True)

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
        self.coarse_transformer = STN(self.cfg.coarse_module)

        # Fourier
        if self.cfg.train:
            beta = self.cfg.fourier_converter.beta_train
        else:
            beta = self.cfg.fourier_converter.beta_test
        self.fourier_converter = FourierConverter(beta)

        # TPS
        self.kornia_tps = KorniaTPS(self.cfg.target_size, self.cfg.grid_size)

        # Losses
        if self.cfg.train:
            self.configure_crit()   

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
        
        refine_mesh = self.coarse_transformer(x_coarse)
        refine_kernel_weight_, refine_affine_weights_ = self.kornia_tps(refine_mesh)
        x_refine = warp_image_tps(x_coarse, 
                                  refine_mesh,
                                  refine_kernel_weight_,
                                  refine_affine_weights_)
        
        # calculate fourier in eval mode only
        x_fft_converted = self.fourier_converter(x_refine) # Denoising for friendly OCR
        
        output = {
            "x_converted": x_fft_converted, 
            "x_refine"   : x_refine,
        }
        return output


    def configure_crit(self):
        self.crit_coarse = Loss(self.cfg.loss.coarse)
        self.crit_refine = Loss(self.cfg.loss.refine)
        if self.cfg.loss.mutual.enable:
            self.crit_mutual = Loss(self.cfg.loss.mutual)

    def on_train_start(self):
        torch.cuda.empty_cache()
        if self.cfg.debug:
            self.trainer.datamodule.data_train.__getitem__(0)
    
    def step(self, batch):
        img = batch["img"]
        reference  = batch["reference"] # ground-truth

        # Fourier Converter
        coarse_mesh = self.coarse_transformer(img)
        coarse_kernel_weight_, coarse_affine_weights_ = self.kornia_tps(coarse_mesh)
        x_coarse = warp_image_tps(img, 
                                  coarse_mesh,
                                  coarse_kernel_weight_,
                                  coarse_affine_weights_)
        
        refine_mesh = self.refine_transformer(x_coarse)
        refine_kernel_weight_, refine_affine_weights_ = self.kornia_tps(refine_mesh)
        x_refine = warp_image_tps(x_coarse, 
                                  refine_mesh,
                                  refine_kernel_weight_,
                                  refine_affine_weights_)

        x_fft_coarse_ = self.fourier_converter(x_coarse.cpu())
        x_fft_refine_ = self.fourier_converter(x_refine.cpu())
        ref_ = self.fourier_converter(reference.cpu())
        
        loss_coarse = self.crit_coarse(x_fft_coarse_.cpu(), ref_.detach().cpu())
        loss_refine = self.crit_refine(x_fft_refine_.cpu(), ref_.detach().cpu())

        if self.cfg.loss.mutual.enable:
            D1 = random_tps(img)
            D2 = random_tps(img)

            d1_mesh = self.coarse_transformer(D1)
            d2_mesh = self.coarse_transformer(D2)
            kernel_weight_1, affine_weights_1 = self.kornia_tps(d1_mesh, d2_mesh) # d1 to d2
            kernel_weight_2, affine_weights_2 = self.kornia_tps(d2_mesh, d1_mesh) # d2 to d1

            wraped_D2 = warp_image_tps(D1, 
                                       d1_mesh, 
                                       kernel_weight_1, 
                                       affine_weights_1)

            wraped_D1 = warp_image_tps(D2, 
                                       d2_mesh, 
                                       kernel_weight_2, 
                                       affine_weights_2)

            loss_mutual = self.crit_mutual(wraped_D1.cpu()*255.0, D1.detach().cpu()*255.0) + \
                          self.crit_mutual(wraped_D2.cpu()*255.0, D2.detach().cpu()*255.0)
            w = self.cfg.loss.mutual.weight
            loss_coarse += w * loss_mutual
        
        loss_ = loss_coarse + loss_refine
        return {
            "coarse": loss_coarse,
            "refine": loss_refine,
            "mutual": loss_mutual * 255.,
            "total" : loss_
        }

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.train_loss.update(loss["total"].item())
        for key in loss.keys():
            self.log(key, loss[key].item(), on_step=True, on_epoch=True, prog_bar=True)
        
        del batch
        return {"loss": loss["total"]}

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): 
            self.trainer.datamodule.data_val.__getitem__(0)
    
    def validation_step(self, batch, batch_idx):
        out = self.forward(batch["img"])
        ref_ = self.fourier_converter(batch["reference"])

        loss = self.crit_coarse(out["x_converted"].detach(), ref_.detach())
        self.val_loss.update(loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        del batch
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        self.log_iter_stats()

    def log_iter_stats(self):
        def gpu_mem_usage():
            """Computes the GPU memory usage for the current device (MB)."""
            mem_usage_bytes = torch.cuda.max_memory_allocated()
            return mem_usage_bytes / 1024 / 1024
        mem_usage = gpu_mem_usage()

        stats = {
            "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
            "train_loss": "%.4f"%(self.train_loss.compute().item()),
            "val_loss": "%.4f"%(self.val_loss.compute().item()),
            "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            "mem": int(np.ceil(mem_usage)),
        }
        self.train_loss.reset()
        self.val_loss.reset()
        logger.info(stats)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }