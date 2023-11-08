import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from . import LOSSES


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def output_sim(score0, score1):
    return (score0.unsqueeze(0).repeat(score1.shape[0], 1) - score1.unsqueeze(1).repeat(1, score0.shape[0])) ** 2
    # return score1.unsqueeze(1).mm(score2.unsqueeze(0))


@LOSSES.register_module('thp')
class TemporalHardPairLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, mask_value=-1):
        super(TemporalHardPairLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == "output":
            self.sim = output_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.mask_value = mask_value

    def forward(self, output, target, **kwargs):
        output = torch.sigmoid(output)
        mask = (target.view(-1) != self.mask_value).nonzero().squeeze()
        hp_values = torch.zeros_like(target.view(-1))
        hp_values[mask] = self.calculate_loss(target.view(-1)[mask], output.view(-1)[mask])
        hp_values = hp_values.reshape((target.shape[0], target.shape[1]))

        thp_loss = 0
        for target_win, output_win, hp_value_win in zip(target, output, hp_values):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            thp_loss += hp_value_win[clip_filter].mean()

        return thp_loss

    def calculate_loss(self, anomalies, output):
        abnormalMask = torch.nonzero(anomalies == 1)
        outputAbnormal = output[abnormalMask].mean(1)
        normalMask = torch.nonzero(anomalies == 0)
        outputNormal = output[normalMask].mean(1)
        # compute image-sentence score matrix
        if outputNormal.shape[0] == 0 or outputAbnormal.shape[0] == 0:
            return torch.zeros_like(output, requires_grad=True)
        scores = self.sim(outputAbnormal, outputNormal)
        # abnormalScores = self.sim(outputAbnormal, outputAbnormal)
        # mask = torch.eye(outputAbnormal.shape[0], outputAbnormal.shape[0]).bool().cuda()
        # abnormalScores.masked_fill_(mask, abnormalScores.min()-5)
        # normalScores = self.sim(outputNormal, outputNormal)
        # mask = torch.eye(outputNormal.shape[0], outputNormal.shape[0]).bool().cuda()
        # normalScores.masked_fill_(mask, normalScores.min()-5)

        # tempAbnormal = outputAbnormal - outputAbnormal[abnormalScores.argmax(dim=0)]
        # tempNormal = outputAbnormal - outputNormal[scores.argmin(dim=0)]
        # abnormalLoss = tempAbnormal**2 - tempNormal**2 + self.margin
        # abnormalLoss = -((outputAbnormal - outputNormal[scores.argmin(dim=0)]) - self.margin)
        losses = torch.zeros_like(output)
        abnormalLoss = -((torch.log(outputAbnormal) - torch.log(outputNormal[scores.argmin(dim=0)])) - self.margin)
        losses[abnormalMask.squeeze()] = abnormalLoss
        # tempNormal = outputNormal - outputNormal[normalScores.argmax(dim=0)]
        # tempAbnormal = outputNormal - outputAbnormal[scores.argmin(dim=1)]
        # normalLoss = tempNormal**2 - tempAbnormal**2 + self.margin
        normalLoss = -((torch.log(outputAbnormal[scores.argmin(dim=1)]) - torch.log(outputNormal)) - self.margin)
        losses[normalMask.squeeze()] = normalLoss

        loss = torch.cat((abnormalLoss, normalLoss), 0)
        filteredDists = torch.max(loss, torch.zeros_like(loss))

        return filteredDists
