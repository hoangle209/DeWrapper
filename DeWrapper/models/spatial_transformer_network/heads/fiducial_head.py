import torch.nn as nn
import torch

from ..blocks.conv import dilation_conv_bn_act

class FiducialHead(nn.Module):
    def __init__(self, n_classes, num_filter, BatchNorm, in_channels=3):
        act_fn = nn.ReLU(inplace=True)
        map_num = [1, 2, 4, 8, 16]
        map_num_i = 3
        self.num_filter = num_filter
        self.bridge_1 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, dilation=1),
            # conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
        )
        self.bridge_2 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=2),
        )
        self.bridge_3 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=5),
        )
        self.bridge_4 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=8),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=3),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=2),
        )
        self.bridge_5 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=12),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=7),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=4),
        )
        self.bridge_6 = nn.Sequential(
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=18),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=12),
            dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                act_fn, BatchNorm, dilation=6),
        )

        self.bridge_concate = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[2], kernel_size=1, stride=1, padding=0),
            # BatchNorm(GN_num, self.num_filter * map_num[4]),
            BatchNorm(self.num_filter * map_num[2]),
            # nn.BatchNorm2d(self.num_filter * map_num[4]),
            act_fn,
        )
        self.out_regress = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),
        )
        self.segment_regress = nn.Linear(self.num_filter * map_num[2]*31*31, 2)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)
    
    def forward(self, x):
        bridge_1 = self.bridge_1(x)
        bridge_2 = self.bridge_2(x)
        bridge_3 = self.bridge_3(x)
        bridge_4 = self.bridge_4(x)
        bridge_5 = self.bridge_5(x)
        bridge_6 = self.bridge_6(x)
        bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concate(bridge_concate)

        out_regress = self.out_regress(bridge)
        segment_regress = self.segment_regress(bridge.view(x.size(0), -1))
        return out_regress, segment_regress