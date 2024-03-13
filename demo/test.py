import sys
sys.path.insert(1, ".")

from omegaconf import DictConfig, OmegaConf
from DeWrapper.datamodules.datamodule import WrapDocDatamodule

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    from DeWrapper.models.de_wrapper import DeWrapper
    import torch
    cfg = OmegaConf.load("config/default.yaml")
    cfg.target_doc_w = 64
    cfg.target_doc_h = 64

    img = torch.ones(1, 3, 64, 64)
    img = img.to("cuda")

    logger.info(f"Instantiating model <{DeWrapper.__name__}>")
    model = DeWrapper(cfg)
    model = model.to("cuda")

    model(img)

