import sys
sys.path.insert(1, ".")

import os
from typing import List

import pyrootutils
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, Timer
from omegaconf import OmegaConf, DictConfig

from DeWrapper.models.spatial_transformer_network.stn import STN
from DeWrapper.datamodules.datamodule import Fiducial2014Datamodule

from DeWrapper.utils import get_pylogger
logger = get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

def train(cfg: DictConfig):
    logger.info(f"Instantiating datamodule <{Fiducial2014Datamodule.__name__}>")
    datamodule = Fiducial2014Datamodule(cfg)

    logger.info(f"Instantiating model <{STN.__name__}>")
    model = STN(cfg)

    logger.info("Instantiating callbacks ...")
    callbacks = [
        TQDMProgressBar(refresh_rate=1), 
        LearningRateMonitor(), 
        Timer()
    ]

    logger.info(f"Instantiating trainer <{Trainer.__qualname__}>")
    cfg_trainer = OmegaConf.to_object(cfg.trainer)
    trainer: Trainer = Trainer(callbacks=callbacks,  **cfg_trainer)
    
    # look for latest checkpoint in logdir and load it if found
    if(cfg.get("ckpt_path") is None):
        path_tmp = cfg.paths.output_dir + "/last.ckpt"
        if(os.path.exists(path_tmp)):
            checkpoint_path = path_tmp
            logger.info("Loading weights from last ckpt " + checkpoint_path)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = cfg.get("ckpt_path")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
        else:
            logger.info("Loading weights from ckpt " + checkpoint_path)

    if cfg.get("train"):
        logger.info("Starting training !!!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path) 


import argparse
parser = argparse.ArgumentParser(
                    prog='Trainer',
                    description='Trainer Augments')

parser.add_argument("--config", type=str, default="config/resnet50_coarse_pretrain.yaml")

if __name__ == "__main__":
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train(config)
