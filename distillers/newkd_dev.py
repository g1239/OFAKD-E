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
        self.dkd_alpha = 1
        self.dkd_beta = 2
        self.dkd_temperature =1
        #assert is_cnn_model(student) , 'current newKD implementation only support cnn student models!'

        self.projector = nn.ModuleDict()
        self.num_projector = 0
        #moe_position_list = [] # distinguish each moe operator
        position = 0
        #moe_position_list = [] # distinguish each moe operator
        for operator in student.modules():
            if hasattr(operator, 'routing_weights_cache'):
                #new_name = name.replace('.','_')
                position +=1
                #projector = nn.Linear(64, self.args.num_classes, bias=True) # TODO pass args.intermediate dimension
                projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(8, self.args.num_classes, bias=False) ,orthogonal_map='cayley') #orthogonal_map='matrix_exp'
                nn.init.zeros_(projector.weight)
                set_module_dict(self.projector, position, projector) #将stage和卷积映射层记录到self.projector字典中
                self.num_projector = self.num_projector + 1


    def forward(self, image, label, epoch, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)
        if len(label.shape) == 2:  # mixup / smoothing
            target = label.max(1)[1]
        else:
            target = label
        
        route_student_losses = []        
        position = 0       
        for operator in self.student.modules():
            if hasattr(operator, 'routing_weights_cache'):
                #new_name = name.replace('.','_')
                '''
                weights_teacher = get_module_dict(self.projector, new_name)(logits_teacher) #将logit交给对应位置的projector处理，输出1*num_expert tensor
                route_student_losses.append( routing_loss( operator.routing_weights_cache,weights_teacher ) )#TODO pass args.moe_temperature
                route_student_cache.append(operator.routing_weights_cache) #debug
                '''
                position +=1
                logits_route = get_module_dict(self.projector, position)(operator.routing_weights_cache) #将logit交给对应位置的projector处理，输出1*num_classes tensor
                #route_student_losses.append( routing_loss( logits_route,logits_teacher ) )#TODO pass args.moe_temperature
                route_student_losses.append(kd_loss(logits_teacher, logits_route, 0.8))#  add reverse KL divergence to stronger supervision

        lossmask_ratio = 0 if epoch < 120 else 1
        loss_kd = self.args.kd_loss_weight * dkd_loss(logits_student, logits_teacher, target,
                                                      self.dkd_alpha, self.dkd_beta, self.dkd_temperature)
        loss_gt = self.args.newkd_gt_loss * self.criterion(logits_student, label)
        loss_route = lossmask_ratio * self.args.newkd_routing_loss / self.num_projector * sum(route_student_losses)  #FIXME select the dim of mean op for safer loss calculation

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

def dist_loss(logits_student, logits_teacher, beta=1., gamma=1., temperature=1.):
    y_s = (logits_student / temperature).softmax(dim=1)
    y_t = (logits_teacher / temperature).softmax(dim=1)
    inter_loss = temperature ** 2 * inter_class_relation(y_s, y_t)
    intra_loss = temperature ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter_loss + gamma * intra_loss


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

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature ** 2)
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature ** 2)
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt