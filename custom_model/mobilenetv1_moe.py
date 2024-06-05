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


__all__ = ['MobileNetV1_moe', 'mobilenetv1_moe']


class DynamicConv(nn.Module): 
    """ Dynamic Conv layer 
        to selective replace the normal Conv2d in relevant method of model class
    """
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=8):
        super().__init__()
        print('+++', num_experts)
        self.num_experts = num_experts
        #self.routing = nn.Linear(in_features, num_experts)
        self.interm_d = 64
        self.proj1 = nn.Linear(in_features, self.interm_d)
        #self.avg_pool = F.avg_pool1d(_ , kernel_size=self.interm_d // num_experts , stride=self.interm_d // num_experts)  # (self.interm_d,here is 256) mod num_experts must be 0 !!! 
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        self.routing_weights_cache = 0
        
    def forward(self, x): # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  
        #new_pooled_inputs = pooled_inputs.detach()
        #new_pooled_inputs.requires_grad_()

        routing_weights_temp = self.proj1(pooled_inputs)
        self.routing_weights_cache = routing_weights_temp # pass self.routing_weights_cache to distiller

        routing_weights = (F.avg_pool1d(torch.sigmoid(routing_weights_temp), kernel_size=self.interm_d // self.num_experts, stride=self.interm_d // self.num_experts))
        x = self.cond_conv(x, routing_weights)
        return x


class MobileNetV1_moe(nn.Module):
    def __init__(self, pretrained_cfg=None, num_classes=1000):
        super(MobileNetV1_moe, self).__init__()

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

        def conv_dw_moe(inp, oup, stride):
            return nn.Sequential(
                DynamicConv(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                DynamicConv(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
       
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),#3
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),#6
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),#9
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw_moe(512, 512, 1), #12
            conv_dw_moe(512, 1024, 2),
            conv_dw_moe(1024, 1024, 1), #14
            nn.AvgPool2d(7),    #15
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        '''
        x = self.model[3][:-1](self.model[0:3](x))
        x = self.model[5][:-1](self.model[4:5](F.relu(x)))
        x = self.model[11][:-1](self.model[6:11](F.relu(x)))
        x = self.model[13][:-1](self.model[12:13](F.relu(x)))
        x = self.model[14](F.relu(x))
        '''
        x = self.model(x)
        x = x.reshape(-1, 1024)
        out = self.fc(x)
        return out


@register_model
def mobilenetv1_moe(pretrained=False, **kwargs):
    model = MobileNetV1_moe(**kwargs)
    model.default_cfg = {'architecture': 'mobilenetv1_moe'}
    return model






