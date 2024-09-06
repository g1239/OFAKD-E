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

        self.projector_list = nn.ModuleDict()
        self.operator_list = nn.ModuleDict()       
        self.num_projector = 0
        position = 0
        #moe_position_list = [] # distinguish each moe operator
        for operator in student.modules():
            if hasattr(operator, 'rwc'):
                #new_name = name.replace('.','_')
                position += 1
                #projector = nn.Linear(64, self.args.num_classes, bias=True) # TODO pass args.intermediate dimension
                #projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(8, self.args.num_classes, bias=False) ,orthogonal_map='cayley') #orthogonal_map='matrix_exp'
                projector = nn.Linear(8, self.args.num_classes,  bias=False) 
                nn.init.zeros_(projector.weight)
                set_module_dict(self.projector_list, position, projector)
                set_module_dict(self.operator_list, position, operator) #将stage和卷积映射层记录到self.projector字典中
                
        self.num_projector = position + 1  #FIXME
            #self.projector.apply(init_weights)


    def forward(self, image, label, epoch, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)
        
        route_student_losses = [] 
        position = 0
        for i in range(1,self.num_projector):
            #new_name = name.replace('.','_')
            '''
            weights_teacher = get_module_dict(self.projector, new_name)(logits_teacher) #将logit交给对应位置的projector处理，输出1*num_expert tensor
            route_student_losses.append( routing_loss( operator.routing_weights_cache,weights_teacher ) )#TODO pass args.moe_temperature
            route_student_cache.append(operator.routing_weights_cache) #debug
            '''

            
            logits_route = get_module_dict(self.projector_list, i)(get_module_dict(self.operator_list, i).rwc) #将logit交给对应位置的projector处理，输出1*num_classes tensor
            #route_student_losses.append( routing_loss( logits_route,logits_teacher ) )#TODO pass args.moe_temperature
            route_student_losses.append(kd_loss(logits_teacher, logits_route, 0.8))#  add reverse KL divergence to stronger supervision

        lossmask_ratio = 0 if epoch < 10 else 1
        loss_kd = self.args.newkd_kd_loss * kd_loss(logits_student, logits_teacher, 1) #TODO pass args.moe_temperature
        loss_gt = self.args.newkd_gt_loss * self.criterion(logits_student, label)
        loss_route = lossmask_ratio * self.args.newkd_routing_loss  * sum(route_student_losses)  #FIXME select the dim of mean op for safer loss calculation / self.num_projector

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
