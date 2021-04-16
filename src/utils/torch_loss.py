import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy(pred, y, num_classes, gamma=2):
    # focal loss
    target = F.one_hot(y, num_classes=num_classes)
    P = F.softmax(pred, dim=1)
    probs = (P*target).sum(dim=1).view(-1 ,1)  # [batch, 1]
    log_p = probs.log()  # [batch, 1]

    loss = -((torch.pow((1-probs), gamma))*log_p).mean()
    return loss


def ohem_loss(rate, cls_pred, cls_target, weight, auxiliary_loss=None):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', weight=weight)

    if auxiliary_loss is not None:
        ohem_cls_loss += auxiliary_loss

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss
