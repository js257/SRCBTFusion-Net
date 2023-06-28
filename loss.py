import torch
import torch.nn as nn
from DualTaskLoss import DualTaskLoss
import torch.nn.functional as F
import numpy as np
##########################################

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
###############################################################

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss
#########################################
from functional import label_smoothed_nll_loss
# from joint_loss import JointLoss
# from dice import DiceLoss
from typing import Optional
from torch import nn, Tensor
__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W)
            target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        nclass = input.shape[1]
        target = one_hot(target.long(), nclass)

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = F.softmax(input, dim=1)
        C = target.shape[1]

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C

###################################################################
class JointEdgeSegLoss(nn.Module):

    def __init__(self, classes=7, dice_weight=0.1, seg_weight=1, att_weight=1, dual_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.dice_weight = dice_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight
        self.nll_loss = nn.NLLLoss2d(ignore_index=255)
        self.multdiceloss = MultiClassDiceLoss().cuda()
        self.dual_loss = DualTaskLoss(cuda=True)
        self.seg_loss = nn.CrossEntropyLoss().cuda()
        self.dic = BinaryDiceLoss().cuda()
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255).cuda()
    def bce2d(self, input, target):
        # c=1
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)


        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)


        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num


        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        loss = 0.0
        filler = torch.ones_like(target) * 255
        targets = torch.where(edge.max(1)[0] > 0.8, target, filler)
        for i in range(0, input.shape[0]):
            loss += self.nll_loss(F.log_softmax(input[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss

    def forward(self, inputs, targets):
        losses = {}
        if self.training and len(inputs) == 2:
            segin, edgin = inputs
            segmask = targets
            losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
            # losses['dice_loss'] = self.dice_weight * self.multdiceloss(segin, segmask)
            # losses['edge_loss'] = self.edge_weight*self.bce_loss(edgin,segmask)
            losses['dual_loss'] = self.dual_weight * self.dual_loss(segin, segmask)
        else:
            segin = inputs
            segmask = targets
            losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)

            # t = F.one_hot(targets, 6)
            # t.permute(0, 3, 1, 2)
            # losses['dice_loss'] = self.dice_weight * self.multdiceloss(segin, segmask)
            # losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
            losses['dual_loss'] = self.dual_weight * self.dual_loss(segin, segmask)

        return losses









