import sys
sys.path.insert(1, ".")

from omegaconf import DictConfig, OmegaConf
from DeWrapper.models.spatial_transformer_network.stn import STN
import torch 

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/coarse_control_points_training.yaml")
    model = STN(cfg.coarse_module)

    im = torch.randn(1, 3, 224, 224)
    out = model(im)
    print(out.size())
    
    
    
