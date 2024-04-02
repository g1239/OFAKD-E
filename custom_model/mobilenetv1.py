import torch.nn as nn
import torch.nn.functional as F
import torch

from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import SelectAdaptivePool2d, Linear, CondConv2d, hard_sigmoid, make_divisible, DropPath
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.helpers import build_model_with_cfg

import math
from functools import partial





__all__ = ['MobileNetV1', 'mobilenetv1']


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), #11
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1), #13
            nn.AvgPool2d(7),    #14
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model[3][:-1](self.model[0:3](x))
        x = self.model[5][:-1](self.model[4:5](F.relu(x)))
        x = self.model[11][:-1](self.model[6:11](F.relu(x)))
        x = self.model[13][:-1](self.model[12:13](F.relu(x)))
        x = self.model[14](F.relu(x))
        x = x.reshape(-1, 1024)
        out = self.fc(x)
        return out


@register_model
def mobilenetv1(pretrained=False, **kwargs):
    model = MobileNetV1()
    model.default_cfg = {'architecture': 'mobilenetv1'}
    return model
