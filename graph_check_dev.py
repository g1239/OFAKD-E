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
from timm import utils
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, Mixup, \
    resolve_data_config
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *

from custom_forward import register_new_forward
from distillers import get_distiller
from utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor
from custom_model import *

import sys
from collections import defaultdict
from types import MethodType

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

visible_devices = "3"  # 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

# load teacher,student and distiller
teacher = create_model(
    "resnet152",
    num_classes=100,
    )
load_checkpoint(
    teacher,
    "/wzy/output/finetune/resnet152-1/checkpoint/model_best.pth.tar",
    )
teacher.requires_grad_(False)
teacher.eval()

student = create_model(
    "resnet18_moe",
    pretrained=False,
    #in_chans=3,
    num_classes=100,
    scriptable=False,
    checkpoint_path=None,
    #**factory_kwargs,
    #**args.model_kwargs,
    )


train_loss_fn = nn.CrossEntropyLoss()
validate_loss_fn = nn.CrossEntropyLoss().cuda()

Distiller = get_distiller("newkd")  
distiller = Distiller(
    student,
    teacher=teacher, 
    criterion=train_loss_fn,
    args=argparse.Namespace(
    num_classes=100,
    kd_temperature=4,
    newkd_kd_loss=1,
    newkd_gt_loss=1,
    newkd_routing_loss=0.05,
    ), 
    #num_data=len(dataset_train)
)
distiller = distiller.cuda()


# screen out the params needed by optimizer2
module_routing_part1 = [name for name, param in distiller.named_parameters() if "projector" in name]
module_routing_part2 = [name for name, ops in student.named_modules() if hasattr(ops, 'routing_weights_cache')]

params_student_part1 = [param for name, param in distiller.named_parameters() if name in module_routing_part1]
params_student_part2 = [param for name, param in student.named_parameters() if name in module_routing_part2]

optimizer1 = create_optimizer_v2(
    distiller, 
    **optimizer_kwargs(
        cfg=argparse.Namespace(
            opt="sgd",
            lr=0.01,
            weight_decay=2e-3,
            momentum=0.9,
        ),
    )
)
optimizer2 = create_optimizer_v2(
    params_student_part1 + params_student_part2, 
    **optimizer_kwargs(
        cfg=argparse.Namespace(
            opt="sgd",
            lr=0.001,
            weight_decay=2e-3,
            momentum=0.9,
        ),
    )
)

# 模拟输入和标签
dummy_input = torch.randn(1, 3, 224, 224).cuda()
target = torch.randint(0, 10, (1,)).cuda()


#teacher_output = teacher(dummy_input).detach()  # 教师网络输出
#student_output = student(dummy_input)  # 学生网络输出
#distillation_loss = train_loss_fn(F.log_softmax(student_output, dim=1), F.softmax(teacher_output, dim=1))

amp_autocast = suppress 

distiller.train()
additional_input = None


with amp_autocast():
    output, losses_dict = distiller(dummy_input, target)
    #loss = sum(losses_dict.values())
    loss = sum(value for key,value in losses_dict.items() if key != "loss_route")
    loss2 = sum(value for key,value in losses_dict.items() if key == "loss_route")

# 优化器1,2更新各自部分参数
optimizer1.zero_grad()
optimizer2.zero_grad()
loss.backward(retain_graph=True)  # 保留计算图以便更新其余参数
loss2.backward()
optimizer1.step()
optimizer2.step()




