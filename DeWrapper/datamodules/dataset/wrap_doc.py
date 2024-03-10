import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import glob

from ..augmentation import RandAugment
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
        self.train = train

        datapath = self.cfg.dataset.path
        self.img_list = glob.glob(f"{datapath}/image/**/*.jpg", recursive=True)
        self.label_list = glob.glob(f"{datapath}/digital/**/*.jpg", recursive=True)
        self.margin_label_list = glob.glob(f"{datapath}/digital_margin/**/*.jpg", recursive=True)

        self.aug = self.augment()
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        path = img_path.split("/")
        path[-3] = "digital"
        ref_path = "/".join(path)
        path[-3] = "digital_margin"
        margin_ref_path = "/".join(path)

        img = pil_loader(img_path)
        ref = pil_loader(ref_path)
        margin_ref = pil_loader(margin_ref_path)

        img = self.aug(img)
        
        return {
            "img": img,
            "ref": ref
        }

    def augment(self):
        min_ = min(self.cfg.target_height, self.cfg.target_width)
        max_ = max(self.cfg.target_height, self.cfg.target_width)
        t = [T.v2.RandomResize(min_, max_)]
        
        rand_aug = self.cfg.dataset.rand_aug
        if self.train and rand_aug is not None:
            if rand_aug == "randaugment":
                t += [T.RandAugment()]
            elif rand_aug == "autoaugment":
                t += [T.AutoAugment()]
            else:
                logger.warning(f"Augment type {rand_aug} is not implemented")

        t += [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std =[0.229, 0.224, 0.225]
            ),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        ]
        return T.Compose(t)
    
    def deform_aug(self):
        t = [
            T.Resize((self.cfg.target_height, self.cfg.target_width)),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std =[0.229, 0.224, 0.225]
            )
        ]







