import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict
from .utils import kd_loss

@register_distiller
class newkd_dev(BaseDistiller):
    requires_feat = False
    require_route = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(newkd_dev, self).__init__(student, teacher, criterion, args)
        self.projector = nn.Linear(8, self.args.num_classes,  bias=False) 
        nn.init.zeros_(self.projector.weight)

        for operator in student.modules():
            if hasattr(operator, 'rw'):
                last_op = operator
        self.last_op = last_op
        
    def forward(self, image, label, epoch, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)       
        logits_route = self.projector(self.last_op.rw)

        lossmask_ratio = 0 if epoch < 80 else 1
        loss_kd = self.args.newkd_kd_loss * kd_loss(logits_student, logits_teacher, 1) #TODO pass args.moe_temperature
        loss_gt = self.args.newkd_gt_loss * self.criterion(logits_student, label)
        loss_route = lossmask_ratio * self.args.newkd_routing_loss  * kd_loss( logits_teacher, logits_route, 0.8)  #FIXME select the dim of mean op for safer loss calculation / self.num_projector

        #torch.set_printoptions(edgeitems=logits_teacher.numel())
        #print(logits_teacher)


     
        losses_dict = {
            "loss_kd": loss_kd,
            "loss_gt": loss_gt,
            "loss_route": loss_route,
        }
    
        return logits_student, losses_dict

def routing_loss(routing_weights, weights_teacher, temperature=1.): #weights_teacher refer to weights calculate by teacher logit
    
    y_t = (weights_teacher / temperature).softmax(dim=1) # use high temperature to soften weights_teacher
   #moe_loss = temperature ** 2 * (1-pearson_correlation(routing_weights, y_t).mean()) #use cosine_similarity to give harder constrain
    return (1-pearson_correlation(routing_weights, y_t).mean()) # get average loss from b*num_proj to 1*num-proj

def js_div(logit_route, logit_teacher, temperature=0.5):
    return 0.5 * kd_loss(logit_route , logit_teacher , temperature=temperature) + 0.5 * kd_loss(logit_teacher , logit_route , temperature=temperature)




def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)

def pearson_correlation(x, y, eps=1e-8): #get pearson_correlation base on cosine_similarity
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)

def dist_loss(logits_student, logits_teacher, beta=1., gamma=1., temperature=1.):
    y_s = (logits_student / temperature).softmax(dim=1)
    y_t = (logits_teacher / temperature).softmax(dim=1)
    inter_loss = temperature ** 2 * inter_class_relation(y_s, y_t)
    intra_loss = temperature ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter_loss + gamma * intra_loss

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()

def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
