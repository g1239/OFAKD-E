import os
from timm.models import create_model
from custom_model import *
import torch
from torchprofile import profile_macs

visible_devices = "3"  # limit visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

model = create_model(
    'resnet50',
    pretrained=False,
    num_classes=100,
    #in_chans=3,
    scriptable=True)

dummy_input = torch.randn(1, 3, 224, 224)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
macs = profile_macs(model, dummy_input)

print('模型可训练参数量：', f"{params/(1e6):.2f}", "M")
print('每样本浮点运算量：', f"{macs/(1e9):.2f}", "G")