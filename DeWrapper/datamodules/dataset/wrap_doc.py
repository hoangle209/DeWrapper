import os
import numpy as np
import torch
import torchvision.transforms as T
import torrchvison.transforms.v2 as T_v2
from PIL import Image
from torch.utils.data import Dataset
import glob

from ..augmentation import SpatialRandAug, ColorRandAug, ClassifyLetterBox
from DeWrapper.utils import make_divisible
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

        datapath = self.cfg.paths.data_dir
        self.img_list = glob.glob(f"{datapath}/image/**/*.jpg", recursive=True)
        self.label_list = glob.glob(f"{datapath}/digital/**/*.jpg", recursive=True)
        self.margin_label_list = glob.glob(f"{datapath}/digital_margin/**/*.jpg", recursive=True)

        self.target_w = self.cfg.target_width
        self.target_h = self.cfg.target_height

        self.configure_aug()
    
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

        input = self.apply_aug_(img, ref)
        return input

    def configure_aug(self):
        aug = []
        colored_aug = []
        ref_aug = []

        # Resize
        if self.cfg.dataset.random_resize:
            min_ = make_divisible(self.target_w*0.75, 32)
            max_ = self.target_w
            aug += [
                T_v2.RandomResize(min_, max_),
                ClassifyLetterBox(size=(self.target_h, self.target_w)), 
                T.ToPILImage()
                ]
        else:
            aug += [T.Resize((self.target_h, self.target_w))]

        # Affine and Color transform
        if self.train and self.cfg.dataset.rand_aug is not None:
            if self.cfg.dataset.rand_aug == "randaugment":
                aug += [SpatialRandAug()]
                colored_aug += [ColorRandAug()]
            elif self.cfg.dataset.rand_aug == "autoaugment":
                colored_aug += [T.AutoAugment()]
            else:
                logger.warning(f"Augment type {self.cfg.dataset.rand_aug} is not implemented")
        
        # To tensor and Normalize
        aug += [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std =[0.229, 0.224, 0.225]
            )]
        colored_aug += [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std =[0.229, 0.224, 0.225]
            )]
        ref_aug += [
            T.Resize((self.target_h, self.target_w)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std =[0.229, 0.224, 0.225]
            )]
        
        # Random Blur
        if torch.rand() < self.cfg.dataset.blur:
            colored_aug += [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))]

        # Random erasing
        if self.cfg.dataset.erasing > 0:
            colored_aug += [
                T.RandomErasing(p=self.cfg.dataset.erasing, inplace=True)
            ]

        self.aug = {
            "nor"    : T.Compsoe(aug),
            "colored": T.Compose(colored_aug),
            "ref"    : T.Compose(ref_aug)
        }

    def apply_aug_(self, img, ref, margin_ref=None):
        img_ = self.aug["nor"](img)
        ref = self.aug["ref"](ref)
        colored = self.aug["colored"](img_)
        deform1 = self.aug["nor"](img)
        deform2 = self.aug["nor"](img)

        return {
            "img"    : img_,
            "ref"    : ref,
            "colored": colored,
            "deform1": deform1,
            "deform2": deform2
        } 
    






