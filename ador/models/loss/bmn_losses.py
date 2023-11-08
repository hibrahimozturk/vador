import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOSSES


@LOSSES.register_module('bmn_box_reg')
class BoxRegLoss(nn.Module):
    def __init__(self):
        super(BoxRegLoss, self).__init__()

    def forward(self, output, target, **kwargs):
        # loss = self.calculate_smooth_l1_loss(output['pred_reg'], target['gt_reg'])
        loss = self.calculate_l1_loss(output['pred_reg'], target['gt_reg'])
        return loss

    @staticmethod
    def calculate_smooth_l1_loss(box_reg_preds, gt_regs, beta=1 / 9):

        losses = []
        for box_reg_pred, gt_reg in zip(box_reg_preds, gt_regs):
            pos_anchors = (gt_reg != 0).all(dim=1)

            if pos_anchors.sum() != 0:
                reg_diff = torch.abs(box_reg_pred[pos_anchors] - gt_reg[pos_anchors])
                regression_loss = torch.where(
                    torch.le(reg_diff, beta),
                    (0.5 * torch.pow(reg_diff, 2)) / beta,
                    reg_diff - 0.5 * beta
                )
                reg_loss = regression_loss.mean()
            else:
                reg_loss = torch.tensor(0.).to(box_reg_pred.device)

            losses.append(reg_loss)

        return torch.stack(losses).mean()

    @staticmethod
    def calculate_l1_loss(box_reg_preds, gt_regs, beta=1 / 9):

        losses = []
        for box_reg_pred, gt_reg in zip(box_reg_preds, gt_regs):
            pos_anchors = (gt_reg != 0).all(dim=1)

            if pos_anchors.sum() != 0:
                reg_diff = torch.abs(box_reg_pred[pos_anchors] - gt_reg[pos_anchors])
                reg_loss = reg_diff.mean()
            else:
                reg_loss = torch.tensor(0.).to(box_reg_pred.device)

            losses.append(reg_loss)

        return torch.stack(losses).mean()


@LOSSES.register_module('bmn_tem')
class TEMLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.1):
        super(TEMLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target, **kwargs):
        loss = self.calculate_loss(output['pred_start'], output['pred_end'], target['gt_start'], target['gt_end'])
        return loss

    def calculate_loss(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = pred_score.view(-1)
            gt_label = gt_label.view(-1)
            pmask = (gt_label > 0.5).float()
            num_entries = len(pmask)
            num_positive = torch.sum(pmask)
            ratio = num_entries / max(1.0, num_positive)
            coef_0 = 0.7 * ratio / (ratio - 1)
            coef_1 = 0.3 * ratio
            epsilon = 0.000001
            # loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
            loss_pos = ((1 - self.alpha) * (1.0 - pred_score) ** self.gamma).detach()\
                       * torch.log(pred_score + epsilon) * pmask
            # loss_pos = torch.log(pred_score + epsilon) * pmask
            # loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            loss_neg = (self.alpha * pred_score ** self.gamma).detach() *\
                       torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            # loss_neg = torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            loss = -1 * torch.sum(loss_pos + loss_neg) / torch.clamp(num_positive, min=1.0)
            # loss = -1 * torch.mean(loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss


@LOSSES.register_module('bmn_pem_reg')
class PEMRegLoss(nn.Module):
    def __init__(self):
        super(PEMRegLoss, self).__init__()

    def forward(self, output, target, **kwargs):
        loss = self.calculate_loss(output['pred_bm'], target['gt_iou_map'], kwargs['bm_mask'])
        return loss

    @staticmethod
    def calculate_loss(pred_bm, gt_iou_map, mask):
        pred_score = pred_bm[:, 0].contiguous()
        gt_iou_map = gt_iou_map * mask

        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = (num_h / max(num_m, 1)) + 0.1
        u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = (num_h / max(num_l, 1)) + 0.1
        u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        # TODO: Attention zero loss for empty windows
        loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / (torch.sum(weights) + 1e-5)

        return loss


class ClsLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.25):
        super(ClsLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_scores, gt_maps):
        losses = []
        for pred_score, gt_iou_map in zip(pred_scores, gt_maps):
            pmask = (gt_iou_map > 0.85).float()
            nmask = (gt_iou_map <= 0.85).float()

            num_positive = torch.sum(pmask)
            # num_entries = num_positive + torch.sum(nmask)
            epsilon = 0.000001
            loss_pos = ((1 - self.alpha) * (1.0 - pred_score) ** self.gamma).detach() * torch.log(pred_score + epsilon) * pmask
            loss_neg = (self.alpha * pred_score ** self.gamma).detach() * torch.log(1.0 - pred_score + epsilon) * nmask
            loss = -1 * torch.sum(loss_pos + loss_neg) / torch.clamp(num_positive, min=1.0)
            losses.append(loss)
        return torch.stack(losses).mean()


@LOSSES.register_module('mid_cls')
class MidClsLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.25):
        super(MidClsLoss, self).__init__()
        self.loss = ClsLoss(gamma, alpha)

    def forward(self, output, target, **kwargs):
        loss = 0
        for i, pred_bm in enumerate(output['pred_bm'][:-1]):
            loss += self.loss(pred_bm, target['gt_iom_map'])

        return loss


@LOSSES.register_module('final_cls')
class FinalClsLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.25):
        super(FinalClsLoss, self).__init__()
        self.loss = ClsLoss(gamma, alpha)

    def forward(self, output, target, **kwargs):
        loss = self.loss(output['pred_bm'][-1], target['gt_iou_map'])
        return loss

    # def calculate_loss(self, pred_scores, gt_iou_maps):
    #     # pred_score = pred_bm[:, 1].contiguous()
    #     # pred_score = pred_bm.contiguous()
    #
    #     # gt_iou_map = gt_iou_map * mask
    #     losses = []
    #     for pred_score, gt_iou_map in zip(pred_scores, gt_iou_maps):
    #         pmask = (gt_iou_map > 0.85).float()
    #         nmask = (gt_iou_map <= 0.85).float()
    #         # nmask = nmask * mask
    #
    #         num_positive = torch.sum(pmask)
    #         num_entries = num_positive + torch.sum(nmask)
    #         ratio = num_entries / max(1.0, num_positive)
    #         coef_0 = 0.7 * ratio / (ratio - 1)
    #         coef_1 = 0.3 * ratio
    #         epsilon = 0.000001
    #         # loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    #         loss_pos = ((1 - self.alpha) * (1.0 - pred_score) ** self.gamma).detach() * torch.log(pred_score + epsilon) * pmask
    #         # loss_pos = torch.log(pred_score + epsilon) * pmask
    #         # loss_pos_ = (torch.log(pred_score + epsilon) * pmask).sum() / num_positive
    #         # loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    #         loss_neg = (self.alpha * pred_score ** self.gamma).detach() * torch.log(1.0 - pred_score + epsilon) * nmask
    #         # loss_neg = torch.log(1.0 - pred_score + epsilon) * nmask
    #         # loss_neg_ = (torch.log(1.0 - pred_score + epsilon) * nmask).sum() / nmask.sum()
    #         loss = -1 * torch.sum(loss_pos + loss_neg) / torch.clamp(num_positive, min=1.0)
    #         # loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    #         losses.append(loss)
    #     return torch.stack(losses).mean()
