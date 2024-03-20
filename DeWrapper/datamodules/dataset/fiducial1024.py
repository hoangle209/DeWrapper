import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset
import glob
import random

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class Fiducial1024(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg