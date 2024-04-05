import sys
sys.path.insert(1, ".")
import cv2

from omegaconf import DictConfig, OmegaConf

from DeWrapper.datamodules.dataset.fiducial1024 import Fiducial1024

if __name__ == "__main__":
    cfg = OmegaConf.load("config/coarse_control_points_training.yaml")
    data = Fiducial1024(cfg)

    dummy = data.__getitem__(0)
    
    p = dummy["fiducial_points"]
    print(p.shape)

