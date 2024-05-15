'''
base on https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

__all__ = ['resnet18_moe', 'resnet34_moe', 'resnet50_moe', 'resnet101_moe',
           'resnet152_moe']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

from timm.models.layers import CondConv2d
class DynamicConv(nn.Module): #only repalce conv1x1
    """ Dynamic Conv layer 
        to selective replace the normal Conv2d in relevant method of model class
    """
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        print('+++', num_experts)
        self.num_experts = num_experts
        #self.routing = nn.Linear(in_features, num_experts)
        self.interm_d = 256
        self.routing = nn.Linear(in_features, self.interm_d)
        #self.avg_pool = F.avg_pool1d(_ , kernel_size=self.interm_d // num_experts , stride=self.interm_d // num_experts)  # (self.interm_d,here is 256) mod num_experts must be 0 !!! 
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        self.routing_weights_cache = 0
        
    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights_temp = torch.sigmoid(self.routing(pooled_inputs))
        self.routing_weights_cache = routing_weights_temp 
        routing_weights = F.avg_pool1d(routing_weights_temp, kernel_size=self.interm_d // self.num_experts, stride=self.interm_d // self.num_experts) 
        x = self.cond_conv(x, routing_weights)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out   

class BasicMoEBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicMoEBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicMoEBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicMoEBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.moe_conv1 = DynamicConv(inplanes, planes, 3, stride) #第三位置变量为kernel_size
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.moe_conv2 = DynamicConv(planes, planes, 3)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.routing_weights_conv1 = 0
        self.routing_weights_conv2 = 0

    def forward(self, x):
        identity = x

        out = self.moe_conv1(x)
        self.routing_weights_conv1=self.moe_conv1.routing_weights_cache
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.moe_conv2(out)
        self.routing_weights_conv2=self.moe_conv2.routing_weights_cache
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def get_routing_weights(self):
        return (self.routing_weights_conv1,self.routing_weights_conv2)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MoEBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MoEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.moe_conv1 = DynamicConv(inplanes, width, 1)
        self.bn1 = norm_layer(width)
        self.moe_conv2 = DynamicConv(width, width, 3, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.routing_weights_conv1 = 0
        self.routing_weights_conv2 = 0

    def forward(self, x):
        identity = x

        out = self.moe_conv1(x)
        self.routing_weights_conv1=self.moe_conv1.routing_weights_cache
        out = self.bn1(out)
        out = self.relu(out)

        out = self.moe_conv2(out)
        self.routing_weights_conv2=self.moe_conv2.routing_weights_cache
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def get_routing_weights(self):
        return self.routing_weights_conv1,self.routing_weights_conv2

class ResNet_moe(nn.Module):

    def __init__(self, denseblock, moeblock, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_moe, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(denseblock, 64, layers[0])
        self.layer2 = self._make_layer(denseblock, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(denseblock, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(moeblock, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * denseblock.expansion, num_classes)    #denseblock与moeblock的expansion系数相同，此处可不作区分

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, MoEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) or isinstance(m, MoEBottleneck):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, denseblock, moeblock, layers, pretrained, progress, **kwargs):
    model = ResNet_moe(denseblock, moeblock, layers, **kwargs)
    assert not pretrained, 'MoE Models have not pretrained weights.'
    return model

@register_model
def resnet18_moe(pretrained=False, pretrained_cfg=None, progress=True, **kwargs):
    model =_resnet('resnet18_moe', BasicBlock, BasicMoEBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
    model.default_cfg = {'architecture': 'resnet18_moe'}
    return model
    

@register_model
def resnet34_moe(pretrained=False, pretrained_cfg=None, progress=True, **kwargs):
    model =_resnet('resnet34_moe', BasicBlock, BasicMoEBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
    model.default_cfg = {'architecture': 'resnet34_moe'}
    return model

@register_model
def resnet50_moe(pretrained=False, pretrained_cfg=None, progress=True, **kwargs):
    model =_resnet('resnet50_moe', Bottleneck, MoEBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
    model.default_cfg = {'architecture': 'resnet50_moe'}
    return model
    

@register_model
def resnet101_moe(pretrained=False, pretrained_cfg=None, progress=True, **kwargs):
    model =_resnet('resnet101_moe', Bottleneck, MoEBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
    model.default_cfg = {'architecture': 'resnet101_moe'}
    return model
    

@register_model
def resnet152_moe(pretrained=False, pretrained_cfg=None, progress=True, **kwargs):
    model =_resnet('resnet152_moe', Bottleneck, MoEBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
    model.default_cfg = {'architecture': 'resnet152_moe'}
    return model



