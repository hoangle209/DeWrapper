from torch import nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=stride, padding=1)


def dilation_conv_bn_act(in_channels, out_dim, act_fn, BatchNorm, dilation=4):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
        BatchNorm(out_dim),
        # nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def dilation_conv(in_channels, out_dim, stride=1, dilation=4, groups=1):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups),
    )
    return model