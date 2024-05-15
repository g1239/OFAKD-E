import torch
import torch.nn.functional as F
from torch import nn
import timm 
import sys
from custom_model import *

model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        
moe_position_list = [] # distinguish each moe operator
#routing_list = [] # cache routing weights
for name, operator in model.named_modules():
    if hasattr(operator, 'qkv'):
        #moe_position_list.append(name)
        #routing_list.append(operator.routing_weights_cache)
        new_name = name.replace('.','_')
        moe_position_list.append(new_name)

print(moe_position_list)
   
