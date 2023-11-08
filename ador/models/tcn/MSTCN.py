import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, feat_dim, in_dim, out_dim, normalize_out=False,
                 output_activation='sigmoid'):
        super(MultiStageModel, self).__init__()
        self.stage_0 = SingleStageModel(num_layers, num_f_maps, feat_dim, out_dim, normalize_out)
        self.output_activation = output_activation
        self.normalize_out = normalize_out
        for s in range(1, num_stages):
            l = copy.deepcopy(SingleStageModel(num_layers, num_f_maps, in_dim, out_dim, normalize_out))
            setattr(self, 'stage_{}'.format(s), l)

        self.stages = nn.ModuleList([getattr(self, 'stage_{}'.format(i)) for i in range(1, num_stages)])

    def forward(self, x, mask):
        x = x.transpose(2, 1)
        mask = mask.transpose(2, 1)
        out, lastLayer = self.stage_0(x, mask)
        # out = out * mask[:, None, :, 0]
        outputs = out.unsqueeze(0)
        for s in self.stages:
            if self.output_activation == 'sigmoid':
                out, lastLayer = s(torch.sigmoid(out), mask)
            elif self.output_activation == 'softmax':
                # output = torch.softmax(out, dim=1)
                # score = ((1-torch.argmax(output, dim=1)) - torch.max(output, dim=1)[0]).__abs__()
                # out, lastLayer = s(score[:, None, :], mask)
                out, lastLayer = s(torch.softmax(out, dim=1), mask)
            # out = out * mask[:, None, :, 0]
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, out_dim, normalize_out):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.normalize_out = normalize_out
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
        # self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)
        self.fc_out = nn.Linear(num_f_maps, out_dim, bias=False)

    def forward(self, x, mask, previousLastLayer=None):
        x = mask[:, None, :, 0] * x
        out = self.conv_1x1(x)
        if previousLastLayer is not None:
            out = out+previousLastLayer
        for layer in self.layers:
            out = layer(out, mask)
        lastLayer = out * mask[:, None, :, 0]
        out_ = out.transpose(1, 2).reshape(-1, out.shape[1])
        # out = self.conv_out(out) * mask[:, 0:1, :]

        if self.normalize_out:
            # self.fc_out = nn.utils.weight_norm(self.fc_out, name='weight', dim=1)
            # for w_order, W in enumerate(self.fc_out.weight):
            #     self.fc_out.weight[w_order] /= torch.norm(self.fc_out.weight[w_order])
                # self.fc_out.weight[w_order] = F.normalize(W, dim=0)

            out_ = F.normalize(out_, dim=1)
            norm_weight = self.fc_out.weight / torch.norm(self.fc_out.weight, dim=1)[:, None]
            out_ = torch.matmul(out_, norm_weight.T)
        else:
            out_ = self.fc_out(out_)

        out = out_.reshape(out.shape[0], out.shape[2], out_.shape[-1]).transpose(1, 2)
        out = out * mask[:, None, :, 0]
        return out, lastLayer


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.batchnorm1 = nn.BatchNorm1d(out_channels)
        # self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.dropout2d = nn.Dropout2d(p=0.3)

    def forward(self, x, mask):
        out = self.conv_dilated(x)
        # out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.conv_1x1(out)
        # out = self.batchnorm2(out)
        # out = self.dropout2d(out.unsqueeze(3))
        out = x + out
        # out = F.relu(out)
        out = out * mask[:, None, :, 0]
        return out


if __name__ == "__main__":
    x = torch.rand(1, 20, 1024).float().cuda()
    mask = torch.ones(1, 1, 20).cuda().float()
    mstcn = MultiStageModel(num_stages=2, num_layers=10, num_f_maps=64, feat_dim=1024, ssRepeat=2)
    mstcn = mstcn.cuda()
    mstcn(x, mask)

    print("finish")
