import sys
sys.path.insert(1, ".")

import os
from typing import List, Optional, Tuple

import lightning as L
import pyrootutils
import submitit
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import OmegaConf, DictConfig

from DeWrapper.models import DeWrapper
from DeWrapper.datamodules.datamodule import WrapDocDatamodule
from DeWrapper.utils.instantiate import instantiate_callbacks, instantiate_loggers
from DeWrapper.utils.ema_checkpoints import EMACheckpoint
from DeWrapper.utils import get_pylogger
logger = get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

def train(cfg: DictConfig):
    logger.info(f"Instantiating datamodule <{WrapDocDatamodule.__name__}>")
    datamodule = WrapDocDatamodule(cfg)

    logger.info(f"Instantiating model <{DeWrapper.__name__}>")
    model = DeWrapper(cfg)

    logger.info("Instantiating callbacks ...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating loggers ...")
    tensornboard_log: List[Logger] = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{Trainer.__qualname__}>")
    cfg_trainer = OmegaConf.to_object(cfg.trainer)
    trainer: Trainer = Trainer(callbacks=callbacks, logger=tensornboard_log, **cfg_trainer)
    
    # look for latest checkpoint in logdir and load it if found
    if(cfg.get("ckpt_path") is None):
        path_tmp = cfg.paths.output_dir + "/last.ckpt"
        path_tmp_ema = cfg.paths.output_dir + "/last-EMA.ckpt"
        if(os.path.exists(path_tmp_ema)):
            checkpoint_path = path_tmp_ema
            logger.info("Loading weights from last EMA ckpt " + checkpoint_path)
        elif(os.path.exists(path_tmp)):
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

    # if cfg.get("train"):
    #     logger.info("Starting training !!!")
    #     trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path) 

if __name__ == "__main__":
    config = OmegaConf.load("config/default.yaml")
    train(config)