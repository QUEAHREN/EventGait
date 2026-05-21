from torch.nn import functional as F
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import torch
import cv2
import numpy as np
import os


block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ResNet9(ResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=3, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True, out_channels=None):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9, self).__init__(block, layers)
        self.layer_end = None
        if out_channels is not None:
            self.layer_end = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer
    




    def forward(self, x):
        x = self.conv1(x)       # in : torch.Size([120, 64, 128, 88])
        x = self.bn1(x)         # in : torch.Size([120, 64, 128, 88])
        x = self.relu(x)        # in : torch.Size([120, 64, 128, 88])
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)      
        x = self.layer2(x)      # in : torch.Size([120, 128, 64, 44])

        x = self.layer3(x)      # in : torch.Size([120, 256, 32, 22])
        x = self.layer4(x)      # in : torch.Size([120, 512, 32, 22])
        if self.layer_end is not None:
            x = self.layer_end(x)
        return x

## 需要一个更浅的resnet用于特征提取
class SimpleResNet(nn.Module):
    def __init__(self, block, channels=[32, 64], in_channel=3, layers=[1, 2], strides=[1, 2], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(SimpleResNet, self).__init__()
        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0])

        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1])

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks >= 1:
            layer = self._make_layer_helper(block, planes, blocks, stride=stride)
        else:
            def layer(x): return x
        return layer

    def _make_layer_helper(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)       # in : torch.Size([24, 3, 64, 64])
        x = self.bn1(x)         # in : torch.Size([24, 64, 64, 64])
        x = self.relu(x)       
        if self.maxpool_flag:
            x = self.maxpool(x) 
        x = self.layer1(x)      # in [24, 64, 32, 32]
        x = self.layer2(x)      # in : torch.Size([24, 64, 32, 32])
        return x



if __name__ == '__main__':
    res = SimpleResNet('BasicBlock', channels=[64, 128], layers=[1, 1], strides=[1, 2], maxpool=True)
    # res = ResNet9('BasicBlock', channels=[64, 128, 256, 512], layers=[1, 1, 1, 1], strides=[1, 2, 2, 1], maxpool=False, other_layer=nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0))
    print(res)
    input = torch.randn(120, 3, 128, 128)
    output, heat = res(input)
    print(output.shape)