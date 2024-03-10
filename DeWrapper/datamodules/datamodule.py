from typing import Any, Dict, Optional

from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .dataset.wrap_doc import WrapDocDataset

class WrapDocDatamodule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train: bool = True,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val  : Optional[Dataset] = None
    
    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            if(self.hparams.train):
                self.data_train = WrapDocDataset(self.hparams.cfg, train=True)
            self.data_val   = WrapDocDataset(self.hparams.cfg, train=False)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.dataloader.batch,
            num_workers=self.hparams.cfg.dataloader.num_workers,
            pin_memory=self.hparams.cfg.dataloader.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.cfg.dataloader.num_workers,
            pin_memory=self.hparams.cfg.dataloader.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    pass