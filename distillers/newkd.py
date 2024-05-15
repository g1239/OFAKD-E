import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict
from .utils import kd_loss

@register_distiller
class newkd(BaseDistiller):
    requires_feat = False
    require_route = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(newkd, self).__init__(student, teacher, criterion, args)

        #assert is_cnn_model(student) , 'current newKD implementation only support cnn student models!'

        self.projector = nn.ModuleDict()
        #moe_position_list = [] # distinguish each moe operator
        #routing_list = [] # cache routing weights
        for name, operator in student.named_modules():
            if 'moe' in name:
                if hasattr(operator, 'routing_weights_cache'):
                    new_name = name.replace('.','_')
                    #projector = nn.Linear(self.args.num_classes, self.args.num_expert, bias=True) #方向为教师logit->路由,尽量减少映射层数
                    projector = nn.Linear(256, self.args.num_classes, bias=True) # TODO pass args.intermediate dimension
                    set_module_dict(self.projector, new_name, projector) #将stage和卷积映射层记录到self.projector字典中
               

        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)
        
        route_student_losses = []        
        route_student_cache = []  #debug
        for name, operator in self.student.named_modules():
            if 'moe' in name:
                if hasattr(operator, 'routing_weights_cache'):
                    new_name = name.replace('.','_')
                    '''
                    weights_teacher = get_module_dict(self.projector, new_name)(logits_teacher) #将logit交给对应位置的projector处理，输出1*num_expert tensor
                    route_student_losses.append( routing_loss( operator.routing_weights_cache,weights_teacher ) )#TODO pass args.moe_temperature
                    route_student_cache.append(operator.routing_weights_cache) #debug
                    '''
                    temp_a = operator.routing_weights_cache.shape
                    temp_b = logits_teacher.shape
                    logits_route = get_module_dict(self.projector, new_name)(operator.routing_weights_cache) #将logit交给对应位置的projector处理，输出1*num_expert tensor
                    temp_c = logits_route.shape
                    route_student_losses.append( routing_loss( logits_route,logits_teacher ) )#TODO pass args.moe_temperature
                    route_student_cache.append(operator.routing_weights_cache) #debug

        #loss_kd = self.args.kd_loss_weight * dist_loss(logits_student, logits_teacher, self.args.dist_beta,
        #                                               self.args.dist_gamma, self.args.dist_tau)
        loss_kd = self.args.newkd_kd_loss * kd_loss(logits_student, logits_teacher, self.args.kd_temperature) #TODO pass args.moe_temperature
        loss_gt = self.args.newkd_gt_loss * self.criterion(logits_student, label)
        loss_route = self.args.newkd_routing_loss * torch.mean(sum(route_student_losses))  

        losses_dict = {
            "loss_kd": loss_kd,
            "loss_gt": loss_gt,
            "loss_route": loss_route,
        }
        return logits_student, losses_dict

def routing_loss(routing_weights, weights_teacher, temperature=1.): #weights_teacher refer to weights calculate by teacher logit
    y_t = (weights_teacher / temperature).softmax(dim=1) # use high temperature to soften weights_teacher
    moe_loss = temperature ** 2 * (1-pearson_correlation(routing_weights, y_t)) #use soft loss to further soften the constrain
    return moe_loss

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
