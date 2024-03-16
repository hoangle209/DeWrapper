import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Timer
from torchmetrics import MeanMetric
from kornia.geometry.transform import remap, warp_image_tps
from kornia.augmentation import RandomThinPlateSpline
import math
import numpy as np

from DeWrapper.models import STN
from DeWrapper.models.utils.fourier_converter import FourierConverter
from DeWrapper.models.utils.thin_plate_spline import KorniaTPS

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

random_tps = RandomThinPlateSpline(scale=0.25, p=1.0, keepdim=True)

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
        logger.info("   - Coarse Transformer...")
        self.cfg.coarse_module.grid_width = self.cfg.grid_width
        self.cfg.coarse_module.grid_height = self.cfg.grid_height
        self.coarse_transformer = STN(self.cfg.coarse_module)

        logger.info("   - Refine Transformer...")
        self.cfg.refine_module.grid_width = self.cfg.grid_width
        self.cfg.refine_module.grid_height = self.cfg.grid_height
        self.refine_transformer = STN(self.cfg.refine_module)

        if self.cfg.train:
            beta = self.cfg.fourier_converter.beta_train
        else:
            beta = self.cfg.fourier_converter.beta_test
        self.fourier_converter = FourierConverter(beta)

        self.kornia_tps = KorniaTPS(self.cfg)

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
        coarse_kernel_weight_, coarse_affine_weights_ = self.kornia_tps(coarse_mesh)
        x_coarse = warp_image_tps(x, 
                                  coarse_mesh,
                                  coarse_kernel_weight_,
                                  coarse_affine_weights_)
        x_coarse = x_coarse[:, :, :self.cfg.target_doc_h, :self.cfg.target_doc_w]
        
        refine_mesh = self.refine_transformer(x_coarse)
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
        loss_type = {
            "L1"       : F.l1_loss,
            "L2"       : F.mse_loss,
            "smooth_L1": F.smooth_l1_loss
        }
        
        # Coarse loss
        crit_coarse = self.cfg.loss.coarse
        if crit_coarse in loss_type:
            self.crit_coarse = loss_type[crit_coarse]
        else:
            logger.warning(f"Coarse loss type {crit_coarse} is not implemented.\
                            Using L1 loss as default...")
            self.crit_coarse = loss_type["L1"]

        # Refinement loss
        crit_refine = self.cfg.loss.coarse
        if crit_refine in loss_type:
            self.crit_refine = loss_type[crit_refine]
        else:
            logger.warning(f"Refinement loss type {crit_refine} is not implemented.\
                            Using L1 loss as default...")
            self.crit_refine = loss_type["L1"]

        # Mutual loss
        if self.cfg.loss.mutual.enable:
            """Document with two different geometric distortions can be 
            mutually transformed to each other if their mesh grids are predicted correctly.
            """
            crit_mutual = self.cfg.loss.mutual.type
            if crit_mutual in loss_type:
                self.crit_mutual = loss_type[crit_mutual]
            else:
                logger.warning(f"Refinement loss type {crit_mutual} is not implemented.\
                                Using L1 loss as default...")
                self.crit_mutual = loss_type["L1"]

    def on_train_start(self):
        torch.cuda.empty_cache()
        if self.cfg.debug:
            self.trainer.datamodule.data_train.__getitem__(0)
    
    def on_train_epoch_end(self):
        logger.info("\n " + self.cfg.paths.output_dir +  " : Training epoch " + str(self.current_epoch) + " ended.")
    
    def step(self, batch):
        normal_img = batch["normal_img"]
        soft_img   = batch["soft_img"]
        hard_img   = batch["hard_img"]
        reference  = batch["reference"] # ground-truth

        # Fourier Converter
        coarse_mesh = self.coarse_transformer(hard_img)
        coarse_kernel_weight_, coarse_affine_weights_ = self.kornia_tps(coarse_mesh)
        x_coarse = warp_image_tps(soft_img, 
                                  coarse_mesh,
                                  coarse_kernel_weight_,
                                  coarse_affine_weights_)
        x_coarse = x_coarse[:, :, :self.cfg.target_doc_h, :self.cfg.target_doc_w]
        del coarse_kernel_weight_, coarse_affine_weights_
        
        refine_mesh = self.refine_transformer(x_coarse)
        refine_kernel_weight_, refine_affine_weights_ = self.kornia_tps(refine_mesh)
        x_refine = warp_image_tps(x_coarse, 
                                  refine_mesh,
                                  refine_kernel_weight_,
                                  refine_affine_weights_)
        del refine_kernel_weight_, refine_affine_weights_

        x_fft_coarse_ = self.fourier_converter(x_coarse.cpu())
        x_fft_refine_ = self.fourier_converter(x_refine.cpu())
        ref_ = self.fourier_converter(reference.cpu())
        
        loss_coarse = self.crit_coarse(x_fft_coarse_, ref_.detach(), reduction='mean')
        loss_refine = self.crit_refine(x_fft_refine_, ref_.detach(), reduction='mean')

        if self.cfg.loss.mutual.enable:
            D1 = random_tps(normal_img)
            D2 = random_tps(normal_img)

            d1_mesh = self.coarse_transformer(D1)
            d2_mesh = self.coarse_transformer(D2)
            kernel_weight_1, affine_weights_1 = self.kornia_tps(d1_mesh, d2_mesh) # d1 to d2
            kernel_weight_2, affine_weights_2 = self.kornia_tps(d2_mesh, d1_mesh) # d2 to d1

            wraped_D2 = warp_image_tps(D1, 
                                       d1_mesh, 
                                       kernel_weight_1, 
                                       affine_weights_1)
            del kernel_weight_1, affine_weights_1

            wraped_D1 = warp_image_tps(D2, 
                                       d2_mesh, 
                                       kernel_weight_2, 
                                       affine_weights_2)
            del kernel_weight_2, affine_weights_2

            loss_mutual = self.crit_mutual(wraped_D1.cpu(), D1.detach().cpu(), reduction='mean') + \
                          self.crit_mutual(wraped_D2.cpu(), D2.detach().cpu(), reduction='mean')
            w = self.cfg.loss.mutual.weight
            loss_coarse += w * loss_mutual * 255.
        
        loss_ = loss_coarse + loss_refine

        del D1, D2
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
        out = self.forward(batch["normal_img"])
        ref_ = self.fourier_converter(batch["reference"].cpu())

        loss = self.crit_coarse(out["x_converted"].cpu(), ref_.detach(), reduction='mean')
        self.val_loss.update(loss.item())
        
        # update and log metrics
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        
        del batch
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        self.log_iter_stats(self.cfg.log_frequency)

    def log_iter_stats(self, cur_iter=1): # TODO
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

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        # linear learning rate scaling for multi-gpu
        if(self.trainer.num_devices * self.trainer.num_nodes>1 and self.cfg.solver.apply_linear_scaling):
            self.lr_scaler = self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches * self.cfg.dataloader.batch / 256
        else:
            self.lr_scaler = 1
        logger.info("num_devices: {}, num_nodes: {}, accumulate_grad_batches: {}, train_batch: {}".format(self.trainer.num_devices, self.trainer.num_nodes, self.trainer.accumulate_grad_batches, self.cfg.dataloader.batch))
        logger.info("Linear LR scaling factor: {}".format(self.lr_scaler))
        
        if(self.cfg.solver.layer_decay is not None):
            optim_params = self.get_param_groups()
        else:
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.solver.lr * self.lr_scaler}]
        
        if(self.cfg.solver.name=="AdamW"):
            optimizer = torch.optim.AdamW(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.95))
        elif(self.cfg.solver.name=="lion"):
            from lion_pytorch import Lion
            optimizer = Lion(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.99))
        elif(self.cfg.solver.name=="SGD"):
            optimizer = torch.optim.SGD(params=optim_params, momentum=self.cfg.solver.momentum, weight_decay=self.cfg.solver.weight_decay)
        else:
            raise NotImplementedError("Unknown solver : " + self.cfg.solver.name)

        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.solver.warmup_epochs:
                lr = (epoch+1) / self.cfg.solver.warmup_epochs
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((epoch+1) - self.cfg.solver.warmup_epochs) / (self.trainer.max_epochs - self.cfg.solver.warmup_epochs )))
            return lr

        if(self.cfg.solver.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.solver.decay_steps, gamma=self.cfg.solver.decay_gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                'frequency': 1,
            }
        }