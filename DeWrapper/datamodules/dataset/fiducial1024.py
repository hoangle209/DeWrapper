import torchvision.transforms as T
from torch.utils.data import Dataset
import glob
import pickle

from ..augmentation.albument import get_appearance_transform


class Fiducial1024(Dataset):
    default_size = [960, 1024]
    fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
    col_gap = row_gap = 1

    def __init__(self, cfg, train=True):
        self.cfg = cfg
        self.train = train
        self.data = glob.glob(f"{cfg.paths.data_dir}/**/color/*.gw", recursive=True)
        self.augments = get_appearance_transform(["visual", "noise", "color"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = self.data[idx]
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        img = data["image"].astype("uint8")
        if self.train:
            img = self.augments(image=img[..., ::-1])["image"]

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.cfg.target_size[0], self.cfg.target_size[1])),
            T.ToTensor()
        ])

        fiducial_points = data["fiducial_points"] / self.default_size * [self.cfg.target_size[1], self.cfg.target_size[0]]
        fiducial_points = 2 * fiducial_points / [self.cfg.target_size[1], self.cfg.target_size[0]] - 1 # convert to range (-1, 1)
        row_gap = self.fiducial_point_gaps[self.row_gap]
        col_gap = self.fiducial_point_gaps[self.col_gap]

        return {
            "img": transform(img),
            "fiducial_points": fiducial_points[::row_gap, ::col_gap].reshape((-1, 2))
        }



