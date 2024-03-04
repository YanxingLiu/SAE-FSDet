from turtle import forward

import torch
# from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.cross_entropy_loss import cross_entropy
from torch.autograd import Variable


class MultiCEFocalLoss(torch.nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)).cuda()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
        ids = target.view(-1, 1)
        # 注意，这里的alpha是给定的一个list(tensor#,里面的元素分别是每一个类的权重因子
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对pt
        log_p = probs.log()  # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class LCCFocalLoss(nn.Module):

    def __init__(self,
                 num_base_classes,
                 num_novel_classes,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super().__init__()
        self.num_base_classes = num_base_classes
        self.num_novel_classes = num_novel_classes
        self.num_classes = self.num_base_classes + self.num_novel_classes + 1
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        if alpha is None:
            self.alpha = Variable(torch.ones(self.num_classes, 1))
        else:
            self.alpha = alpha

        self.cls_criterion = cross_entropy
        self.cls_criterion1 = MultiCEFocalLoss(
            num_novel_classes + 1, reduction=self.reduction)

    def forward(self,
                predict,
                predict_novel,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # if self.class_weight is not None:
        #     class_weight = predict.new_tensor(
        #         self.class_weight, device=predict.device)
        # else:
        #     class_weight = None
        label_stage0 = torch.clamp(target, 0, self.num_base_classes)
        loss_cls_stage0 = self.loss_weight * self.cls_criterion(
            predict,
            label_stage0,
            weight,
            class_weight=None,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=None,
            avg_non_ignore=False,
            **kwargs)

        label_stage1_index = target >= self.num_base_classes
        cls_score_stage1 = predict_novel[label_stage1_index]
        label_stage1 = target[label_stage1_index] - self.num_base_classes
        # weight_stage1 = weight[label_stage1_index]
        # avg_factor_stage1 = float(len(cls_score_stage1))
        loss_cls_stage1 = self.loss_weight * \
            self.cls_criterion1(cls_score_stage1, label_stage1)

        return dict(loss_cls_s0=loss_cls_stage0, loss_cls_s1=loss_cls_stage1)
