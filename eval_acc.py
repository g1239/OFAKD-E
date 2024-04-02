import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 加载预训练模型
model = timm.create_model('resnet50', pretrained=True)

# 准备评估数据
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root='/wzy/dataset/imagenet-1k/ILSVRC/Data/CLS-LOC', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 将模型设置为评估模式
model.eval()

top1_correct = 0
top5_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = outputs.topk(5, 1, largest=True, sorted=True)  # 获取Top-5预测
        total_samples += labels.size(0)
        labels = labels.view(-1, 1)
        top1_correct += predicted[:, 0].eq(labels).sum().item()  # 计算Top-1准确率
        top5_correct += predicted.eq(labels).sum().item()  # 计算Top-5准确率

top1_accuracy = top1_correct / total_samples
top5_accuracy = top5_correct / total_samples

print("Top-1 Accuracy:", top1_accuracy)
print("Top-5 Accuracy:", top5_accuracy)
