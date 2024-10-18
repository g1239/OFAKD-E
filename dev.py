import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.utils
import yaml
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, Mixup, \
    resolve_data_config
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from custom_forward import register_new_forward
from distillers import get_distiller
from utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor
from custom_model import *


visible_devices = "3"  # limit visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]

model = create_model(
    'resnet18_moe',
    pretrained=False,
    num_classes=100,
    #in_chans=3,
    scriptable=True)
projector_list = nn.ModuleDict()
operator_list = nn.ModuleDict()   
position = 0
for name,operator in model.named_modules():
    if hasattr(operator, 'rwc'): #and name.endswith('1.moe_conv2'):
        #print((name))
        set_module_dict(operator_list, position, operator)
        
        #print(operator)
        #print(name)
        #print(operator     )                   
#print(model.layer1[0])
#print(model.layer1[1].moe_conv1.rwc)

         
position = 0

dummy_input = torch.randn(1, 3, 224, 224)
model(dummy_input)
'''
for operator in model.modules():
    if hasattr(operator, 'rwc'):
        
        position +=1
        #projector = nn.Linear(8, 4, bias=False) 
        #nn.init.zeros_(projector.weight)
        #set_module_dict(projector_list, position, projector)
        set_module_dict(operator_list, position, operator)
        print((operator).rwc) 
'''
#print(model)
#print(model.layer1[0].moe_conv1.cond_conv.weight.shape)
a = torch.flatten(torch.svd(model.layer4[0].moe_conv1.cond_conv.weight).U)
print(a.shape)
print(model.layer4[0].moe_conv1.rwc.shape)
s =  model.layer4[0].moe_conv1.rwc  *a
print(s.shape)


#print(s.shape)
#print(v.shape)
#print(projector_list)
#for i in range(1,position+1):
    #print(get_module_dict(projector_list, position))
    #print(get_module_dict(operator_list, i))
    #print(get_module_dict(projector_list, position)(get_module_dict(operator_list, position).rwc))
#print(operator_list)




