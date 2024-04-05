import torch.nn as nn
from ..blocks.conv import *

class ResidualBlockWithDilated(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 BatchNorm,  
                 kernel_size=3,
                 stride=1,
                 downsample=None, 
                 is_activation=True, 
                 is_top=False, 
                 is_dropout=False):
        super().__init__()
        self.stride = stride
        self.is_activation = is_activation
        self.downsample = downsample
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilation_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilation_conv(out_channels, out_channels, kernel_size, dilation=3)		# 3
        
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetV2StraightV2(nn.Module):
    def __init__(self, 
                 num_filter=32, 
                 map_num=[1, 2, 4, 8, 16], 
                 BatchNorm="batch", 
                 block_nums=[3, 4, 6, 3], 
                 block=ResidualBlockWithDilated, 
                 stride=[1, 2, 2, 2], 
                 dropRate=[0.2, 0.2, 0.2, 0.2], 
                 kernel_size=5,
                 is_sub_dropout=False):
        
        super().__init__()
        self.channels = []
        self.in_channels = num_filter * map_num[0]
        self.dropRate = dropRate
        self.stride = stride
        self.is_sub_dropout = is_sub_dropout
        self.relu = nn.ReLU(inplace=True)
        self.kernel_size = kernel_size

        if BatchNorm == "group":
            BatchNorm = nn.GroupNorm
        elif BatchNorm == "batch":
            BatchNorm = nn.BatchNorm2d
        elif BatchNorm == "instance":
            BatchNorm = nn.InstanceNorm2d

        self.resnet_head = nn.Sequential(
            nn.Conv2d(3, num_filter * map_num[0], kernel_size=3, stride=2, padding=1),
            BatchNorm(num_filter * map_num[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filter * map_num[0], num_filter * map_num[0], kernel_size=3, stride=2, padding=1),
            BatchNorm(num_filter * map_num[0]),
            nn.ReLU(inplace=True),
        )

        self.block_nums = block_nums
        self.layer1 = self.blocklayer(
                                block, 
                                num_filter * map_num[0], 
                                self.block_nums[0], 
                                BatchNorm, 
                                kernel_size, 
                                stride=self.stride[0]
                            )
        self.layer2 = self.blocklayer(
                                block, 
                                num_filter * map_num[1], 
                                self.block_nums[1], 
                                BatchNorm, 
                                kernel_size, 
                                stride=self.stride[1]
                            )
        self.layer3 = self.blocklayer(
                                block, 
                                num_filter * map_num[2], 
                                self.block_nums[2], 
                                BatchNorm, 
                                kernel_size, 
                                stride=self.stride[2]
                            )
        # self.layer4 = self.blocklayer(
        #                         block, 
        #                         num_filter * map_num[3], 
        #                         self.block_nums[3], 
        #                         BatchNorm, 
        #                         kernel_size, 
        #                         stride=self.stride[3])

        self.strides = 16
        self._initialize_weights()
        

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, kernel_size, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                BatchNorm(out_channels))

        layers = []
        layers.append(block(
                        self.in_channels, 
                        out_channels, 
                        BatchNorm, 
                        kernel_size, 
                        stride, 
                        downsample, 
                        is_top=True, 
                        is_dropout=False
                        ))
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(block(
                            out_channels, 
                            out_channels, 
                            BatchNorm, 
                            kernel_size, 
                            is_activation=True, 
                            is_top=False, 
                            is_dropout=self.is_sub_dropout
                            ))
            
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
        x = self.resnet_head(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        # out4 = self.layer4(out3)
        return out3

    


