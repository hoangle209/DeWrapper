import torch
import torch.nn as nn
from lightning import LightningModule

from .backbones.builder import builder as backbone_builder
from .heads.builder import builder as head_builder

from DeWrapper.models.losses.create_loss import Loss
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class STN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.backbone = backbone_builder(self.cfg)
        self.cfg.backbone.out_channel = self.backbone.channels[-1]
        self.head = head_builder(self.cfg)
        self.act = nn.Hardtanh()

        self.configure_loss()
    
    def forward(self, x):
        x = self.act(self.head(self.backbone(x)))
        return x

    def train_step(self, batch, batch_idx):
        source_points = self.forward(batch["img"]) # source points
        loss = self.crit(source_points, batch["source_points"])
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}
        
    def validation_step(self, batch, batch_idx):
        source_points = self.forward(batch["img"]) # source points
        loss = self.crit(source_points, batch["source_points"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
 
    def configure_loss(self):
        self.crit = Loss(self.cfg.loss.coarse)
    
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
        

        