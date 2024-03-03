from torch import nn

class FDRHead(nn.Module):
    """Fourier Document Restoration Network Head
    This head is implemented from: https://arxiv.org/abs/2203.09910
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
