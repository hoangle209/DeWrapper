import sys
sys.path.insert(1, ".")

from omegaconf import DictConfig, OmegaConf
from DeWrapper.models.de_wrapper import DeWrapper
from DeWrapper.models.spatial_transformer_network.stn import STN
import cv2
import torch 

from DeWrapper.models.icdar2021 import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/resnet50_coarse_pretrain.yaml")
    model = STN(cfg.coarse_module)
    # model = DeWrapper.load_from_checkpoint("weights/last.ckpt", map_location="cpu", cfg=cfg)
    # model.eval()

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)  
    # cv2.resizeWindow("img", 650, 1000)

    # img = cv2.imread("images/0000.jpg")
    # img = cv2.resize(img, (768, 1088))
    
    # img = img.transpose(2, 0, 1)[None] / 255.0
    # img = torch.from_numpy(img)

    # out = model(img)
    # img = out["x_converted"]

    # img = img[0].cpu().numpy()
    # img = img.transpose(1, 2, 0).astype("uint8")

    # print(img.shape)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    
    
    
