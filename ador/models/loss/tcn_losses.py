import torch.nn as nn
import torch
from . import LOSSES


@LOSSES.register_module('l1')
class TCNL1Loss(nn.Module):
    def __init__(self, mask_value=-1):
        super(TCNL1Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mask_value = mask_value

    def forward(self, output, target, **kwargs):
        output = torch.sigmoid(output)
        l1_loss = 0
        for target_win, output_win in zip(target, output):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            output_win = output_win.squeeze()
            l1_loss += self.l1_loss(output_win[clip_filter], target_win[clip_filter])
        return l1_loss


@LOSSES.register_module('bce')
class TCNBCELoss(nn.Module):
    def __init__(self, mask_value=-1):
        super(TCNBCELoss, self).__init__()
        self.mask_value = mask_value
        self.loss = torch.nn.BCELoss()

    def forward(self, output, target, **kwargs):
        output = torch.sigmoid(output)
        # mse_loss = 0
        mse_loss = []
        for target_win, output_win in zip(target, output):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            output_win = output_win.squeeze()
            # mse_loss += self.losses.bce.loss(output_win[clip_filter], target_win[clip_filter])
            mse_loss.append(self.loss(output_win[clip_filter], target_win[clip_filter]))
        mse_loss = torch.mean(torch.stack(mse_loss))
        return mse_loss


@LOSSES.register_module('mse')
class TCNMSELoss(nn.Module):
    def __init__(self, mask_value=-1):
        super(TCNMSELoss, self).__init__()
        self.mask_value = mask_value
        self.loss = torch.nn.MSELoss()

    def forward(self, output, target, **kwargs):
        output = torch.sigmoid(output)
        mse_loss = 0
        for target_win, output_win in zip(target, output):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            output_win = output_win.squeeze()
            mse_loss += self.loss(output_win[clip_filter], target_win[clip_filter])
        return mse_loss

