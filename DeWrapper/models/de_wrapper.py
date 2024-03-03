from torch import nn
from DeWrapper.utils import get_pylogger
logger = get_pylogger()

class DeWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg