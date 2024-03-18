import sys
sys.path.insert(1, ".")

from omegaconf import OmegaConf
from DeWrapper.models.utils.fourier_converter import FourierConverter
import cv2
import torch

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    fft = FourierConverter(0.008)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)  
    # # cv2.resizeWindow("img", 650, 1000)
    img = cv2.imread("C:\\Users\\ADMIN\\Downloads\\Dataset\\WarpDoc\\train\\digital\\rotate\\0157.jpg")
    img = cv2.resize(img, (768, 1088))
    
    img = img.transpose(2, 0, 1)[None]
    img = torch.from_numpy(img)
    print(isinstance(img, torch.Tensor))

    img = fft(img.float(), False)
    img = img[0].cpu().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")

    print(img.shape)

    cv2.imshow("img", img)
    cv2.waitKey(0)