import sys
sys.path.insert(1, ".")

from omegaconf import DictConfig, OmegaConf
from DeWrapper.datamodules.datamodule import WrapDocDatamodule

from DeWrapper.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    from DeWrapper.models.de_wrapper import DeWrapper
    cfg = OmegaConf.load("config/default.yaml")
    # model = DeWrapper(cfg)
    # oimg = torch.ones(1, 3, 224, 512)
    # gimg = np.ones((224, 512), dtype=np.uint8)
    # img = torch.ones(1, 3, 512, 224)
    # i = {
    #     "origin_img": oimg,
    #     "gray_img": gimg,
    #     "img": img,        
    # }
    # model(img)
    # model.eval()
    # print(model.training)
    
    logger.info(f"Instantiating datamodule <{WrapDocDatamodule.__name__}>")
    datamodule = WrapDocDatamodule(cfg)
    datamodule.setup()
    # train_dataloader = datamodule.train_dataloader()
    datamodule.data_train.__getitem__(0)
