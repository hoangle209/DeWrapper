from .resnet import resnet_builder

def builder(cfg):
    name = cfg.backbone.type
    if "resnet" in name:
        backone = resnet_builder(name)
    
    return backone