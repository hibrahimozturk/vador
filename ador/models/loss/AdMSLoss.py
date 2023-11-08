import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOSSES


@LOSSES.register_module('admsoftmax')
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, s=30.0, m=0.4, mask_value=-1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.mask_value = mask_value

    def forward(self, output, target, **kwargs):
        # output = torch.softmax(output, dim=1)
        output = output.transpose(1, 2)
        mse_loss = []
        # mse_loss = 0
        for target_win, output_win in zip(target, output):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            mse_loss.append(self.calculate_loss(output_win[clip_filter], target_win[clip_filter]))
        mse_loss = torch.mean(torch.stack(mse_loss))
        return mse_loss

    def calculate_loss(self, wf, labels):
        '''
        input shape (N, in_features)
        '''
        # assert len(x) == len(labels)
        # assert torch.min(labels) >= 0
        # assert torch.max(labels) < self.out_features

        # for W in self.fc.parameters():
        #     W = F.normalize(W, dim=1)
        #
        # x = F.normalize(x, dim=1)
        #
        # wf = self.fc(x)
        labels = labels.long()
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        numerator_ = torch.diagonal(wf.transpose(0, 1)[labels])
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        denominator_ = torch.exp(numerator_) + torch.sum(torch.exp(excl), dim=1)
        L = numerator - torch.log(denominator)
        L_ = numerator_ - torch.log(denominator_)
        return -torch.mean(L)
