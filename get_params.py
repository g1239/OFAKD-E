import torch
import timm
import sys
sys.path.append('../custom_model')
from custom_model import *
# 实例化目标timm模型
model = timm.create_model('resnet', pretrained=False)

# 计算可学习参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params)


