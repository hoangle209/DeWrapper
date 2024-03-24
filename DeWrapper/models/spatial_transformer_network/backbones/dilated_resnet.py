import torch.nn as nn
from ..blocks.conv import *

class ResidualBlockWithDilatedV1(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, stride=1, downsample=None, is_activation=True, is_top=False, is_dropout=False):
        super(ResidualBlockWithDilatedV1, self).__init__()
        self.stride = stride
        self.is_activation = is_activation
        self.downsample = downsample
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, self.stride)
        else:
            self.conv1 = dilation_conv(in_channels, out_channels, dilation=3)		# 3
        self.bn1 = BatchNorm(out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.stride != 1 or self.is_top:
            self.conv2 = conv3x3(out_channels, out_channels)
        else:
            self.conv2 = dilation_conv(out_channels, out_channels, dilation=3)		# 1
        self.bn2 = BatchNorm(out_channels)

        self.is_dropout = is_dropout
        self.drop_out = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        # if self.is_dropout:
        # 	out1 = self.drop_out(out1)
        out = self.bn2(self.conv2(out1))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNetV2StraightV2(nn.Module):
    def __init__(self, 
                 num_filter, 
                 map_num, 
                 BatchNorm, 
                 block_nums=[3, 4, 6, 3], 
                 block=ResidualBlockWithDilatedV1, 
                 stride=[1, 2, 2, 2], 
                 dropRate=[0.2, 0.2, 0.2, 0.2], 
                 is_sub_dropout=False):
        
        super().__init__()
        self.channels = []
        self.in_channels = num_filter * map_num[0]
        self.dropRate = dropRate
        self.stride = stride
        self.is_sub_dropout = is_sub_dropout
        self.drop_out = nn.Dropout2d(p=dropRate[0])
        self.drop_out_2 = nn.Dropout2d(p=dropRate[1])
        self.drop_out_3 = nn.Dropout2d(p=dropRate[2])
        self.drop_out_4 = nn.Dropout2d(p=dropRate[3]) 	
        self.relu = nn.ReLU(inplace=True)

        self.block_nums = block_nums
        self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, stride=self.stride[0])
        self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, stride=self.stride[1])
        self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, stride=self.stride[2])
        self.layer4 = self.blocklayer(block, num_filter * map_num[3], self.block_nums[3], BatchNorm, stride=self.stride[3])

        self._initialize_weights()

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                BatchNorm(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, BatchNorm, stride, downsample, is_top=True, is_dropout=False))
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(block(out_channels, out_channels, BatchNorm, is_activation=True, is_top=False, is_dropout=self.is_sub_dropout))
            
        self.channels.append(out_channels)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x, is_skip=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4

    


