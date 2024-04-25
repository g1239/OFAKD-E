import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import timm
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
#from distillers import get_distiller
from utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor
from custom_model import *

import sys
from collections import defaultdict
from types import MethodType


a = torch.tensor([1.,2.,3.],requires_grad=True)
class DyConv(nn.Module): #only repalce conv1x1
    """ Dynamic Conv layer 
        to selective replace the normal Conv2d in relevant method of model class
    """
    def __init__(self, input):
        super().__init__()
        self.routing_weights_cache = input.clone()
             
        
        print(self.routing_weights_cache.data_ptr())
       

print(a.data_ptr())
b = DyConv(a)