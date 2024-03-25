import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset
import glob
import random
from ..augmentation.letterbox import ClassifyLetterBox

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
            
        self.target_w, self.target_h = self.cfg.target_size
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
        resize = v2.Resize((self.target_h, self.target_w))
        leeterbox_resize = ClassifyLetterBox(size=(self.target_h, self.target_w))
        
        to_tensor_and_norm = [
            T.ToTensor(),
            T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ]

        # Blur
        blur = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))

        # Geometry
        p_geometry = 1 - self.cfg.dataset.geometry
        geometry = [
            v2.RandomRotation((-30, 30)), 
            v2.RandomPerspective(distortion_scale=0.3, p=p_geometry), 
        ]

        color_jiter = T.ColorJitter(0.2, 0.2, 0.2, 0.2)
        
        self.aug = {
            "resize": resize,
            "letterbox_reisize": leeterbox_resize,
            "to_tensor_and_norm": T.Compose(to_tensor_and_norm),
            "geometry": T.Compose(geometry),
            "blur": blur,
            "color_jiter": color_jiter
        }

    def apply_aug_(self, img, ref):
        input = {}
        img = self.aug["letterbox_reisize"](img)
        if self.train: 
            if random.random() > (1 - self.cfg.dataset.color_jiter):
                img = self.aug["color_jiter"](img)

        input |= {
            "img"       : self.aug["to_tensor_and_norm"](img),
            "reference" : self.aug["to_tensor_and_norm"](self.aug["resize"](ref)),
        }
    
        return input 