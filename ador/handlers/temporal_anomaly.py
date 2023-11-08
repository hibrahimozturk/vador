from handlers.temporal_action import TemporalActionSegHandler
from abc import ABCMeta, abstractmethod
# from models.loss.temporal_hard_pair import TemporalHardPairLoss
# from models.loss.AdMSLoss import AdMSoftmaxLoss
# from models.TALNet.BMN import bmn_loss_func, pem_cls_loss_func, pem_reg_loss_func, tem_loss_func
# from models.TALNet.BMN_v3 import box_reg_loss_func
# from models.loss.bmn_losses import BoxRegLoss, TEMLoss, PEMRegLoss, PEMClsLoss
from addict import Dict
import torch
import numpy as np
import copy
from models.loss import LOSSES

# from dataset.ucf_crime import UCFCrimeTemporal, collate_fn_precomp
# from torch.utils.data import DataLoader


class TemporalAnomalyDetectionHandler(TemporalActionSegHandler):
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        TemporalActionSegHandler.__init__(self, cfg)
        self.num_classes = 2
        self.thresholds = cfg.evaluation.thresholds

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    @abstractmethod
    def model_forward(self, data, evaluate=False, **kwargs):
        pass

    # def get_dataloaders(self, data_cfg, mode):
    #
    #     if mode == "train":
    #         train_dataset = UCFCrimeTemporal(data_cfg, split="train")
    #
    #         train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
    #                                   num_workers=data_cfg.num_workers,
    #                                   collate_fn=collate_fn_precomp,
    #                                   pin_memory=True)
    #
    #         val_dataset = UCFCrimeTemporal(data_cfg, split="val")
    #
    #         val_loader = DataLoader(val_dataset, batch_size=data_cfg.batch_size, shuffle=False,
    #                                 num_workers=data_cfg.num_workers,
    #                                 collate_fn=collate_fn_precomp)
    #
    #         return train_loader, val_loader
    #
    #     elif mode == "test":
    #         test_dataset = UCFCrimeTemporal(data_cfg, split="test")
    #
    #         test_loader = DataLoader(test_dataset, batch_size=data_cfg.batch_size, shuffle=False,
    #                                  num_workers=data_cfg.num_workers,
    #                                  collate_fn=collate_fn_precomp)
    #         return test_loader

    def filter_data(self, data):
        mask = data["anomalies"] != self.mask_value
        for key in data:
            data[key] = data[key][mask]
        return data

    def clip_outputs(self, output, clip_targets, clip_names):
        clip_data = Dict()
        clip_data.anomalies = clip_targets.view(-1).detach().cpu().numpy()
        clip_data.clip_names = np.array(clip_names).reshape(-1)
        clip_data.outputs = output.view(-1).detach().cpu().numpy()

        clip_data = self.filter_data(clip_data)
        clip_data.clip_names = clip_data.clip_names.tolist()

        return clip_data

    # def visualize_outputs(self, video_clips, exp_dir):
    #     pass

    @staticmethod
    def append_overlapped_clips(clip_data):
        clip_dicts = Dict(predictions=dict(),
                          targets=dict())

        for clip_name, prediction, target in zip(clip_data.clip_names, clip_data.outputs,
                                                 clip_data.anomalies):
            if clip_name not in clip_dicts.predictions:
                clip_dicts.predictions[clip_name] = []
            if clip_name not in clip_dicts.targets:
                clip_dicts.targets[clip_name] = []
            clip_dicts.predictions[clip_name].append(float(prediction))
            clip_dicts.targets[clip_name].append(float(target))
        return clip_dicts

    @staticmethod
    def create_losses(loss_cfg):
        losses = Dict()
        for loss_dict in loss_cfg:
            if hasattr(loss_dict, 'params'):
                loss = LOSSES.get(loss_dict.type)(**loss_dict.params)
            else:
                loss = LOSSES.get(loss_dict.type)()
            losses[loss_dict.type] = Dict(loss=loss,
                                          factor=loss_dict.factor)

        return losses

    @staticmethod
    def move_2_gpu(data):
        input_data, masks, anomalies, category, _, clip_names = data
        # obj_features, obj_boxes, masks, anomalies, category, _, clip_names = data
        gpu_data = Dict()
        if torch.cuda.is_available():
            gpu_data.input = input_data.float().cuda()
            gpu_data.anomalies = anomalies.float().cuda()
            gpu_data.masks = masks.float().cuda()
        gpu_data.clip_names = clip_names
        return gpu_data

    def temporal_score(self, iou_list, video_clips, bg_class=0):
        threshold_scores = Dict()
        for threshold in self.thresholds:
            c_video_clips = copy.deepcopy(video_clips)
            for video_name in c_video_clips:
                c_video_clips[video_name].predictions = (np.array(c_video_clips[video_name].predictions) > threshold).tolist()
            output_scores = super().temporal_score(iou_list, c_video_clips, bg_class=bg_class)
            threshold_scores["thr_{:.2f}".format(threshold)] = output_scores
        return threshold_scores

    def calculate_loss(self, output, target, **kwargs):
        total_loss = 0
        loss_outputs = dict()
        for loss_type, loss_cfg in self.losses.items():
            partial_loss = loss_cfg.loss(output, target, **kwargs)
            loss_outputs[loss_type] = partial_loss

            factor = torch.tensor(float(loss_cfg.factor), requires_grad=True).float()
            if torch.cuda.is_available():
                factor = factor.cuda()
            total_loss += factor * partial_loss
        return total_loss, loss_outputs
