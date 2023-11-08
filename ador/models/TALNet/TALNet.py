# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils_.interp1d import Interp1d, pool_data
from utils_.drop_path import DropPath


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return self.a_2[None, :, None] * (x - mean) / (std + self.eps) + self.b_2[None, :, None]


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, dilation=1, stride=1, groups=1, norm_affine=False,
                 act='relu', norm=True, no_residual=False, drop_path=0.0):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                              padding=padding, groups=groups, stride=stride, dilation=dilation)
        if norm:
            self.norm = nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=False)
            # self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        self.drop_path = DropPath(drop_path)
        if no_residual:
            self.residual_conn = False
        else:
            test_input = torch.randn(1, in_channels, 128)
            test_output = self.conv(test_input)
            if test_output.shape == test_input.shape:
                self.residual_conn = True
            else:
                self.residual_conn = False

    def forward(self, input_x):
        x = self.conv(input_x)
        if x.shape[2] > 1:
            x = self.norm(x)
        x = self.act(x)
        if self.residual_conn:
            x = self.drop_path(x) + input_x
        return x


class Classifier(nn.Module):
    def __init__(self, feat_dim, num_conv=3):
        super(Classifier, self).__init__()

        self.convs = nn.Sequential(
            ConvNormAct(feat_dim, feat_dim, 3, padding=1, no_residual=True),
            ConvNormAct(feat_dim, feat_dim, 3, padding=1, no_residual=True),
            ConvNormAct(feat_dim, 1, 3, padding=1, act='sigmoid', norm=False),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Regression(nn.Module):
    def __init__(self, feat_dim, num_conv=3):
        super(Regression, self).__init__()

        self.convs = nn.Sequential(
            ConvNormAct(feat_dim, feat_dim, 3, padding=1, no_residual=True),
            ConvNormAct(feat_dim, feat_dim, 3, padding=1, no_residual=True),
            ConvNormAct(feat_dim, 2, 3, padding=1, act='identity', norm=False),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class FPN(nn.Module):
    def __init__(self, feat_dim, base_dim, first_time, fpn_levels):
        super(FPN, self).__init__()

        self.first_time = first_time
        if self.first_time:
            fpn_down_channel = []
            for _ in range(fpn_levels):
                fpn_down_channel.append(ConvNormAct(base_dim, feat_dim, 1, 0))

            self.fpn_down_channel = nn.ModuleList(fpn_down_channel)

        self.upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample_2x = nn.MaxPool1d(3, stride=2, padding=1)

        fpn_conv_up = []
        for _ in range(fpn_levels-1):
            fpn_conv_up.append(ConvNormAct(feat_dim, feat_dim, 3, 1, no_residual=True))
        self.fpn_conv_up = nn.ModuleList(fpn_conv_up)

        fpn_conv_down = []
        for _ in range(fpn_levels-1):
            fpn_conv_down.append(ConvNormAct(feat_dim, feat_dim, 3, 1, no_residual=True))
        self.fpn_conv_down = nn.ModuleList(fpn_conv_down)

    def forward(self, inputs):
        if self.first_time:
            input_features = []
            for input_feat, down_channel in zip(inputs, self.fpn_down_channel):
                input_feat = down_channel(input_feat)
                input_features.append(input_feat)
            inputs = input_features

        up_features = [inputs[len(inputs)-1]]

        for feat_index in range(len(inputs)-2, -1, -1):
            up_feature = self.upsample_2x(up_features[-1])
            up_feature = self.fpn_conv_up[feat_index](inputs[feat_index] + up_feature)
            up_features.append(up_feature)

        up_features = list(reversed(up_features))
        down_features = [up_features[0]]
        for feat_index in range(1, len(up_features)):
            down_feature = self.downsample_2x(down_features[-1])
            if feat_index == len(up_features) - 1:
                down_feature = self.fpn_conv_down[feat_index-1](down_feature + inputs[feat_index])
            else:
                down_feature = self.fpn_conv_down[feat_index-1](down_feature + up_features[feat_index] + inputs[feat_index])
            down_features.append(down_feature)

        return down_features


class TALNet(nn.Module):
    """
    Bidirectional FPN, positive anchors for classification are activated based on intersection over min area,
     positive anchors for regression are activated based on iou
    """

    def __init__(self, feat_dim=512, fpn_levels=6,
                 fpn_repeat=2):
        super(TALNet, self).__init__()
        self.feat_dim = feat_dim
        self.fpn_repeat = fpn_repeat

        # self.hidden_dim_1d = 256
        self.base_dim = 256
        self.fpn_dim = 128
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        dpr = [x.item() for x in torch.linspace(0, 0.2, 4)]

        # Base Module
        self.x_1d_b = nn.Sequential(
            ConvNormAct(self.feat_dim, self.base_dim, kernel=1, padding=0),
            *[ConvNormAct(self.base_dim, self.base_dim, kernel=3, padding=1, drop_path=dpr[_]) for _ in range(4)],
        )

        fpn = nn.ModuleList([ConvNormAct(self.base_dim, self.base_dim, 3, 1, no_residual=True)])

        for _ in range(fpn_levels-1):
            fpn.append(ConvNormAct(self.base_dim, self.base_dim, 3, 1, stride=2))

        self.fpn_conv = fpn

        self.fpns = nn.ModuleList(
            [FPN(self.fpn_dim, self.base_dim, _ == 0, fpn_levels) for _ in range(self.fpn_repeat)]
        )
        self.x_1d_confidences = nn.ModuleList([Classifier(self.fpn_dim) for _ in range(self.fpn_repeat)])
        # self.x_1d_confidence = Classifier(self.fpn_dim)

        self.x_1d_regressions = nn.ModuleList([Regression(self.fpn_dim) for _ in range(self.fpn_repeat)])

    def init_head(self):
        prior = 0.01
        # prior = 0.1
        # self.x_1d_s[-1].conv.weight.data.fill_(0)
        # self.x_1d_s[-1].conv.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.x_1d_e[-1].conv.weight.data.fill_(0)
        # self.x_1d_e[-1].conv.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # for block in self.x_1d_confidence:
        for x1d_conf in self.x_1d_confidences:
            x1d_conf.convs[-1].conv.weight.data.fill_(0)
            x1d_conf.convs[-1].conv.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # self.x_1d_confidence.convs[-1].conv.weight.data.fill_(0)
        # self.x_1d_confidence.convs[-1].conv.bias.data.fill_(-math.log((1.0 - prior) / prior))

        for x1d_reg in self.x_1d_regressions:
            x1d_reg.convs[-1].conv.weight.data.fill_(0)
            x1d_reg.convs[-1].conv.bias.data.fill_(0)

    def forward(self, x):
        base_feature = self.x_1d_b(x)

        stage_anchor_scores = []
        stage_anchor_reg = []
        # fpn_features = []
        level_features = []
        for down_scale in self.fpn_conv:
            base_feature = down_scale(base_feature)
            level_features.append(base_feature)
            # fpn_feature = down_channel(base_feature)
            # fpn_features.append(fpn_feature)

        # up_features = [fpn_features[len(fpn_features)-1]]
        #
        # for feat_index in range(len(fpn_features)-2, -1, -1):
        #     up_feature = self.upsample_2x(up_features[-1])
        #     up_feature = self.fpn_conv_up[feat_index](fpn_features[feat_index] + up_feature)
        #     up_features.append(up_feature)

        # for level_feature in reversed(up_features):
        # fpn_features = self.fpns(level_features)
        fpn_features = level_features
        for fpn_order, fpn in enumerate(self.fpns):
            fpn_features = fpn(fpn_features)
            anchor_scores = []
            anchor_reg = []
            for level, level_feature in enumerate(fpn_features):
                # scores = self.x_1d_confidences[level](level_feature)
                scores = self.x_1d_confidences[fpn_order](level_feature)
                regression = self.x_1d_regressions[fpn_order](level_feature)
                anchor_scores.append(scores)
                anchor_reg.append(regression)

            anchor_scores = torch.cat(anchor_scores, axis=2)[:, 0, :]
            anchor_reg = torch.cat(anchor_reg, axis=2)
            stage_anchor_scores.append(anchor_scores)
            stage_anchor_reg.append(anchor_reg)
        return stage_anchor_scores, stage_anchor_reg


if __name__ == '__main__':
    # import opts

    # opt = opts.parse_opt()
    # opt = vars(opt)
    model = TALNet(64)
    input = torch.randn(2, 512, 128)
    a, b = model(input)
    print(a.shape, b.shape)
