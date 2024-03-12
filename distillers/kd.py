import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import kd_loss


@register_distiller
class KD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(KD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():  #禁用局部代码块中的梯度计算
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)         #gt损失由train.py读取训练命令中参数确定，默认为交叉熵损失
        loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)   #由utils.py中实现
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


@register_distiller
class BKD(BaseDistiller):   #与kd类唯一区别在于蒸馏损失基于二值交叉熵
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(BKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * F.binary_cross_entropy_with_logits(logits_student, logits_teacher.softmax(1))
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
