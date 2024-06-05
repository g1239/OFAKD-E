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

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义教师网络
class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 定义学生网络
class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 15)
        self.layer2 = nn.Linear(15, 10)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 一些不属于学生网络的参数（例如某个外部参数）
class ExternalParameters(nn.Module):
    def __init__(self):
        super(ExternalParameters, self).__init__()
        self.external_param = nn.Parameter(torch.randn(10))

teacher_net = TeacherNetwork()
student_net = StudentNetwork()
external_params = ExternalParameters()

# 定义损失函数
criterion = nn.CrossEntropyLoss()
distillation_loss_fn = nn.KLDivLoss()

# 将学生网络的参数根据名称分组
params_student_part1 = [param for name, param in student_net.named_parameters() if 'layer1' in name]
params_student_part2 = [param for name, param in student_net.named_parameters() if 'layer2' in name]
params_external = list(external_params.parameters())

# 创建优化器
optimizer1 = optim.Adam(params_student_part1, lr=0.001)
optimizer2 = optim.Adam(params_student_part2 + params_external, lr=0.001)

# 模拟输入和标签
input_data = torch.randn(10, 10)
target = torch.randint(0, 10, (10,))

# 前向传播
teacher_output = teacher_net(input_data).detach()  # 教师网络输出
student_output = student_net(input_data)  # 学生网络输出

# 计算损失
task_loss = criterion(student_output, target)
distillation_loss = distillation_loss_fn(F.log_softmax(student_output, dim=1), F.softmax(teacher_output, dim=1))

# 总损失
total_loss = task_loss + distillation_loss

# 优化器1更新学生网络部分参数
optimizer1.zero_grad()
total_loss.backward(retain_graph=True)  # 保留计算图以便更新其余参数
optimizer1.step()

# 优化器2更新学生网络其余参数和外部参数
optimizer2.zero_grad()
total_loss.backward()
optimizer2.step()

