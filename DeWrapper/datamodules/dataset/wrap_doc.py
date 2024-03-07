import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import glob


from DeWrapper.utils import get_pylogger
logger = get_pylogger(__name__)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class WrapDocDataset(Dataset):
    def __init__(self, cfg, train=True):
        self.cfg = cfg

        datapath = self.cfg.dataset.path
        self.img_list = glob.glob(f"{datapath}/image/**/*.jpg", recursive=True)
        self.label_list = glob.glob(f"{datapath}/digital/**/*.jpg", recursive=True)
        self.margin_label_list = glob.glob(f"{datapath}/digital_margin/**/*.jpg", recursive=True)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        return 






