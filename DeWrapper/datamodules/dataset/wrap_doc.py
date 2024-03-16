import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T_v2
from PIL import Image
import cv2 
from torch.utils.data import Dataset
import glob
import random

from DeWrapper.datamodules.augmentation.blur import GaussianBlur, MotionBlur, DefocusBlur, ZoomBlur, GlassBlur
from DeWrapper.datamodules.augmentation.geometry import Perspective, TranslateX, TranslateY
from DeWrapper.utils import get_pylogger
logger = get_pylogger(__name__)

DEFAULT_MEAN = [0., 0., 0.]
DEFAULT_STD  = [1., 1., 1.]

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
    ext = ["*.jpg", "*.png"]
    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.train = train

        _t = "train" if train else "val"
        datapath = self.cfg.paths.data_dir
        self.img_list = []
        for e in self.ext:
            self.img_list.extend(glob.glob(f"{datapath}/{_t}/image/**/{e}", recursive=True))

        self.target_w = self.cfg.target_width
        self.target_h = self.cfg.target_height
        self.target_doc_w = self.cfg.target_doc_w
        self.target_doc_h = self.cfg.target_doc_h

        self.configure_aug()
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        path = img_path.split("/")
        path[-3] = "digital"
        ref_path = "/".join(path)
        path[-3] = "digital_margin"

        img = pil_loader(img_path)
        try:
            ref = pil_loader(ref_path)
        except:
            _ext = ref_path[-3:].upper()
            ref_path = ref_path.split('.')
            ref_path[-1] = _ext
            ref_path = '.'.join(ref_path)
            ref = pil_loader(ref_path)

        input = self.apply_aug_(img, ref)
        return input

    def configure_aug(self):
        resize = T.Resize((self.target_h, self.target_w))
        to_tensor_and_norm = [
            T.ToTensor(),
            T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ]

        # Blur
        p_blur = 1 - self.cfg.dataset.blur
        blur = [
            GaussianBlur(prob=p_blur), 
            MotionBlur(prob=p_blur), 
            DefocusBlur(prob=p_blur), 
            ZoomBlur(prob=p_blur), 
            GlassBlur(prob=p_blur)
        ]

        # Geometry
        p_geometry = 1 - self.cfg.dataset.geometry
        geometry = [
            T.RandomRotation((-30, 30), expand=True), 
            Perspective(prob=p_geometry), 
            TranslateX(prob=p_geometry), 
            TranslateY(prob=p_geometry)
        ]

        ref_aug = [
            T.Resize((self.target_doc_h, self.target_doc_w)),
            T.ToTensor(),
            T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ]
        
        self.aug = {
            "resize": resize,
            "to_tensor_and_norm": to_tensor_and_norm,
            "reference": T.Compose(ref_aug), 
            "geometry": T.Compose(geometry),
            "blur": blur,
        }

    def apply_aug_(self, img, ref):
        input = {}
        if self.train:
            img_soft = self.aug["geometry"](self.aug["resize"](img))   
            img_hard = img_soft

            # brightness | constrast | shaepness
            if random.random() > (1 - self.cfg.dataset.brightness):
                brightness_factor = random.uniform(0.9, 1.2)
                img_hard = T.adjust_brightness(img_hard, brightness_factor)
            
            if random.random() > (1 - self.cfg.dataset.constrast):
                constrast_factor = random.uniform(0.9, 1.2)
                img_hard = T.adjust_constrast(img_hard, constrast_factor)

            if random.random() > (1 - self.cfg.dataset.sharpness):
                sharpness_factor = random.uniform(1.0, 1.3)
                img_hard = T.adjust_constrast(img_hard, sharpness_factor)

            # blur
            blur_idx = random.randint(0, len(self.aug["blur"]))
            img_hard = self.aug["blur"][blur_idx](img_hard)

            input |= {
                "soft_img": self.aug["to_tensor_and_norm"](img_soft), 
                "hard_img": self.aug["to_tensor_and_norm"](img_hard)}

        input |= {
            "normal_img": self.aug["to_tensor_and_norm"](self.aug["resize"](img)),
            "reference" : self.aug["ref"](ref),
        }
    
        return input 
    






