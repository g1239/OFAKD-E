import torch
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR100,CIFAR10
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from torch_cka import CKA
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser
import os
from custom_model import *

visible_devices = "3"  # 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


#seed = 42 # set random seed for reproducibility
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = create_model(
    'resnet10_moe',
    pretrained=False,
    num_classes=100,
    #in_chans=3    
    )
load_checkpoint(model1,'/wzy/output/train/t5-21/checkpoint/model_best.pth.tar')

model2 = create_model(
    'resnet10_moe',
    pretrained=False,
    num_classes=100,
    #in_chans=3    
    )
load_checkpoint(model2,'/wzy/output/train/t5-21/checkpoint/model_best.pth.tar')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

batch_size = 256

dataset = CIFAR100(root='/wzy/dataset/cifar100',
                  train=False,
                  download=False,
                  transform=transform)

# 获取测试集的总长度
test_size = len(dataset)

subset_indices = random.sample(range(test_size), test_size // 10)

# 创建测试集的子集
cifar100_test_subset = Subset(dataset, subset_indices)


dataloader1 = DataLoader(dataset,#cifar100_test_subset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

# cka = CKA(model1, model2,
#         model1_name="ResNet18", model2_name="ResNet34",
#         device='cuda')
#
# cka.compare(dataloader)
#
# cka.plot_results(save_path="../assets/resnet_compare.png")


#===============================================================

print("model1: ", model1)

model1.eval()
model2.eval()
cka = CKA(model1, model2,
        model1_name="R10", model2_name="R10-fc",

        
        model1_layers =['layer1.0.moe_conv1', 'layer1.0.moe_conv2',
                        'layer2.0.moe_conv1', 'layer2.0.moe_conv2',
                        'layer3.0.moe_conv1', 'layer3.0.moe_conv2',
                        'layer4.0.moe_conv1', 'layer4.0.moe_conv2'                      
                       ],
        model2_layers =['layer1.0.moe_conv1', 'layer1.0.moe_conv2',
                        'layer2.0.moe_conv1', 'layer2.0.moe_conv2',
                        'layer3.0.moe_conv1', 'layer3.0.moe_conv2',
                        'layer4.0.moe_conv1', 'layer4.0.moe_conv2'                      
                       ],
        device='cuda')

#print(model1.layer1[0].moe_conv1)

cka.compare(dataloader1)
cka.plot_results(save_path="/wzy/output/picture/test2.png")




'''
model1_layers =['layer1.0.moe_conv1', 'layer1.0.moe_conv2',
                        'layer2.0.moe_conv1', 'layer2.0.moe_conv2',
                        'layer3.0.moe_conv1', 'layer3.0.moe_conv2',
                        'layer4.0.moe_conv1', 'layer4.0.moe_conv2'                      
                       ],
        model2_layers =['layer1.0.moe_conv1', 'layer1.0.moe_conv2',
                        'layer2.0.moe_conv1', 'layer2.0.moe_conv2',
                        'layer3.0.moe_conv1', 'layer3.0.moe_conv2',
                        'layer4.0.moe_conv1', 'layer4.0.moe_conv2'                      
                       ],

################################################


model1_layers =['layer1.0.conv1', 'layer1.0.conv2',
                'layer2.0.conv1', 'layer2.0.conv2',
                'layer3.0.conv1', 'layer3.0.conv2',
                'layer4.0.conv1', 'layer4.0.conv2'                      
                ],
        model2_layers =['fc',                     
                ],
'''