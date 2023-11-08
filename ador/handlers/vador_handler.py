import random

from handlers.handler import HANDLERS
from handlers.temporal_anomaly import TemporalAnomalyDetectionHandler

from models.vador.VADOR import VADOR
import numpy as np
from scipy import sparse

from addict import Dict
import torch
import torch.nn as nn
import itertools
import math
import os.path as osp
import metrics
from utils_.utils import visualize_temporal_action_bmn_
from utils_.anchors1d import Anchors1D
import pickle

# from tqdm import autonotebook


@HANDLERS.register_module(name="vador")
class ADORBMNV4Handler(TemporalAnomalyDetectionHandler):
    """
    Box prediction is made by regression head in BMN, instead of start, end predictions
    """

    def __init__(self, cfg):
        TemporalAnomalyDetectionHandler.__init__(self, cfg)

        self.proposal_threshold = cfg.evaluation.proposal_threshold
        self.anchors1d = Anchors1D(anchor_scale=cfg.model.talnet.anchor_scale,
                                   scale_ranges=cfg.model.talnet.scale_ranges,
                                   pyramid_levels=list(range(cfg.model.talnet.fpn_levels)))

        self.snms_alpha = cfg.evaluation.soft_nms.alpha
        self.snms_t1 = cfg.evaluation.soft_nms.low_threshold
        self.snms_t2 = cfg.evaluation.soft_nms.high_threshold

        self.stage_keys = ['relation_transformer']
        self.bm_mask = None

        self.temp_ratio = []
        self.bbox_transform = BBoxTransform()

    def get_model(self, model_cfg):
        model = VADOR(model_cfg)
        return model

    def model_params(self, **kwargs):
        self.model.epoch = kwargs['epoch']

    def grad_clip_norm(self, **kwargs):
        stages = [self.model.relation_transformer, self.model.mstcn.stage1]
        stages += list(self.model.mstcn.stages)
        active_params = []
        for stage_index, stage in enumerate(stages):
            if self.active_stages[stage_index] <= kwargs['epoch']:
                active_params.append(stage.parameters())
        params = itertools.chain(*active_params)
        torch.nn.utils.clip_grad_norm(params, 0.1)

    def prepare_target_maps(self, anomalies, masks, tscale):
        # iou_maps, start_scores, end_scores = [], [], []
        iom_maps, iou_maps, regression_targets = [], [], []
        anchors, scales = self.anchors1d(anomalies)

        anchor_widths = anchors[0, :, 1] - anchors[0, :, 0]
        anchor_ctr_x = anchors[0, :, 0] + 0.5 * anchor_widths

        # temporal_gap = 1. / tscale
        # anchor_xmin = [temporal_gap * (i - 0.5) for i in range(tscale)]
        # anchor_xmax = [temporal_gap * (i + 0.5) for i in range(tscale)]

        for anomaly_map in anomalies:
            # gt_bbox = []
            bboxes = []
            labels, starts, ends = metrics.get_labels_start_end_time(anomaly_map)
            for start, end in zip(starts, ends):
                # gt_bbox.append([start / len(anomaly_map), end / len(anomaly_map)])
                bboxes.append([start, end])

            bboxes = np.array(bboxes)
            # gt_bbox = np.array(gt_bbox)

            if len(bboxes) != 0:
                widths = bboxes[:, 1] - bboxes[:, 0]
                widths = widths[:, None].repeat(len(scales[0]), axis=1)
                repeated_scales = scales.cpu().numpy().repeat(len(bboxes), axis=0)
                range_mask = np.logical_and(repeated_scales[..., 0] < widths, widths < repeated_scales[..., 1])

                gt_iou_map = self.iou_with_anchors(anchors[0, :, 0].cpu().numpy(), anchors[0, :, 1].cpu().numpy(),
                                                   bboxes[:, 0], bboxes[:, 1])

                gt_ioa_map = self.iom_with_anchors(anchors[0, :, 0].cpu().numpy(), anchors[0, :, 1].cpu().numpy(),
                                                   bboxes[:, 0], bboxes[:, 1])

                gt_iou_map[np.logical_not(range_mask)] = 0
                gt_ioa_map[np.logical_not(range_mask)] = 0

                gt_iou_map = torch.Tensor(gt_iou_map)
                gt_ioa_map = torch.Tensor(gt_ioa_map)
                iou_max, iou_argmax = gt_iou_map.max(dim=0)
                iom_max, _ = gt_ioa_map.max(dim=0)
                selected_boxes = bboxes[iou_argmax, :]
                gt_ioa_map = iom_max >= 0.5
                gt_iou_map = iou_max >= 0.5

                targets = torch.zeros([anchors.shape[1], 2]).to(anomalies.device)
                if gt_iou_map.sum() > 0:
                    selected_boxes = selected_boxes[gt_iou_map]

                    gt_widths = selected_boxes[:, 1] - selected_boxes[:, 0]
                    gt_ctr_x = selected_boxes[:, 0] + 0.5 * gt_widths

                    gt_widths = torch.from_numpy(gt_widths).to(anomalies.device)
                    gt_ctr_x = torch.from_numpy(gt_ctr_x).to(anomalies.device)

                    gt_widths = torch.clamp(gt_widths, min=1)
                    targets_dx = (gt_ctr_x - anchor_ctr_x[gt_iou_map]) / anchor_widths[gt_iou_map]
                    targets_dw = torch.log(gt_widths / anchor_widths[gt_iou_map])

                    targets[gt_iou_map] = torch.stack([targets_dx.float(), targets_dw.float()]).t()

            else:
                # gt_iou_map = torch.zeros([tscale, tscale])
                gt_ioa_map = torch.zeros(anchors.shape[1])
                gt_iou_map = torch.zeros(anchors.shape[1])
                targets = torch.zeros((anchors.shape[1], 2)).to(anomalies.device)
                match_score_start = torch.zeros(tscale)
                match_score_end = torch.zeros(tscale)

            iom_maps.append(gt_ioa_map)
            iou_maps.append(gt_iou_map)
            regression_targets.append(targets)

        # start_scores = torch.stack(start_scores).to(anomalies.device)
        # end_scores = torch.stack(end_scores).to(anomalies.device)
        regression_targets = torch.stack(regression_targets)
        iom_maps = torch.stack(iom_maps).to(anomalies.device)
        iou_maps = torch.stack(iou_maps).to(anomalies.device)
        return iom_maps.float(), iou_maps.float(), regression_targets

    def init_eval_epoch_dict(self):
        epoch_dict = super().init_eval_epoch_dict()
        epoch_dict.start_scores = Dict()
        epoch_dict.end_scores = Dict()
        return epoch_dict

    def eval_iteration(self, data, epoch_dict, epoch):
        report, epoch_dict, clip_dicts = super().eval_iteration(data, epoch_dict, epoch)

        epoch_dict.start_scores.update(clip_dicts[-1].start_scores)
        epoch_dict.end_scores.update(clip_dicts[-1].end_scores)

        return report, epoch_dict, clip_dicts

    def organize_video_clip(self, epoch_dict, level=None):
        predictions = epoch_dict.predictions
        targets = epoch_dict.targets

        video_clips = dict()

        entire_predictions = []
        entire_targets = []

        normal_video_preds = []

        dataloader = getattr(self, "{}_loader".format(self.mode))
        video_clip_list = dataloader.dataset.__getVideoClips__()
        video_info_list = dataloader.dataset.__get_video_info__()
        for video_name, clip_list in video_clip_list.items():
            clip_list = clip_list[::1]
            clips = Dict(predictions=[], targets=[], start_scores=[], end_scores=[])
            for clip_name in clip_list:
                clip_pred = np.mean(np.array(predictions[clip_name]))
                clip_gt = np.mean(np.array(targets[clip_name]))

                if 'Normal' in video_name:
                    normal_video_preds.append(clip_pred)

                entire_predictions.append(clip_pred)
                entire_targets.append(clip_gt)

                clips.predictions.append(clip_pred)
                clips.targets.append(clip_gt)

                clips.start_scores.append(epoch_dict.start_scores[clip_name][0])
                clips.end_scores.append(epoch_dict.end_scores[clip_name][0])

            video_info = video_info_list[video_name]
            temporal_gt = np.zeros(video_info['num_frames'])
            fps = video_info['fps']
            for action in video_info['annotations']:
                temporal_gt[math.floor(action['start'] * fps):math.floor(action['end'] * fps)] = 1
                # temporal_gt[math.floor(action['start']):math.floor(action['end'])] = 1

            interp_pred = np.interp(np.linspace(0, len(clips.predictions), video_info['num_frames']),
                                    np.arange(0, len(clips.predictions)), np.array(clips.predictions))

            interp_start = np.interp(np.linspace(0, len(clips.predictions), video_info['num_frames']),
                                     np.arange(0, len(clips.predictions)), np.array(clips.start_scores))

            interp_end = np.interp(np.linspace(0, len(clips.predictions), video_info['num_frames']),
                                   np.arange(0, len(clips.predictions)), np.array(clips.end_scores))

            clips.predictions = interp_pred
            clips.targets = temporal_gt
            clips.start_scores = interp_start
            clips.end_scores = interp_end

            video_clips[video_name] = clips

        return video_clips, entire_predictions, entire_targets, normal_video_preds

    @staticmethod
    def temporal_visualize(video_clips, image_dir, pickels_dir, file_ext):
        output_json = dict()
        for video_name, clips in video_clips.items():
            save_path = osp.join(image_dir, video_name.split(".")[0] + file_ext)
            output_json[video_name] = dict(predictions=(np.array(clips.predictions) * 500).astype(np.uint8),
                                           targets=np.array(clips.targets, dtype=np.bool))
            visualize_temporal_action_bmn_(clips.predictions, clips.targets, clips.start_scores, clips.end_scores,
                                          save_path, video_name)
        with open(osp.join(pickels_dir, 'clips{}.pickle'.format(file_ext.split('.')[0])), 'wb') as fp:
            pickle.dump(output_json, fp)

    def model_forward(self, data, evaluate=False, **kwargs):
        device = torch.device("cuda", self.cfg.proc_id)
        data = self.move_2_gpu(data, device)
        report = dict()
        stage_clip_dicts = None
        loss_dict = {"{}_loss".format(loss_type): [] for loss_type, loss_cfg in self.losses.items()}

        if evaluate:
            iom_maps, iou_maps, regression_targets = self.prepare_target_maps(data.anomalies[:, 0::1],
                                                                              data.masks[:, 0::1],
                                                                              len(data.anomalies[0][0::1]))
            confidence_maps = []
            box_regressions = []
            for i in range(1):
                obj_feats = data["obj_features"][:, i::1]
                obj_boxes = data["obj_boxes"][:, i::1]
                frame_feats = data["frame_features"][:, i::1]
                c_m, b_r = self.model(obj_feats, obj_boxes, frame_feats, data.masks[:, i::1])
                confidence_maps.append(torch.stack(c_m))
                box_regressions.append(torch.stack(b_r))
            confidence_map = torch.stack(confidence_maps).mean(0)
            box_regression = torch.stack(box_regressions).mean(0)
            box_regression = box_regression.transpose(2, 3)
        else:
            iom_maps, iou_maps, regression_targets = self.prepare_target_maps(data.anomalies, data.masks,
                                                                              len(data.anomalies[0]))
            confidence_map, box_regression = self.model(data["obj_features"], data["obj_boxes"],
                                                        data['frame_features'], data.masks)

            for i, _ in enumerate(box_regression):
                box_regression[i] = box_regression[i].transpose(1, 2)
        # box_regression = box_regression.transpose(1, 2)
        output = dict(pred_bm=confidence_map, pred_reg=box_regression[-1])
        target = dict(gt_iom_map=iom_maps, gt_iou_map=iou_maps, gt_reg=regression_targets)
        loss, loss_outputs = self.calculate_loss(output, target, bm_mask=self.bm_mask)

        for loss_type, loss_cfg in self.losses.items():
            loss_dict['{}_loss'.format(loss_type)] = loss_outputs[loss_type]

        loss_dict['total_loss'] = loss
        report['total_loss'] = loss

        if evaluate:
            confidence_map = confidence_map[-1]
            box_regression = box_regression[-1]
            # self.temp_ratio.append(len(data.anomalies[0] / t_scale))
            clip_data = Dict(anomalies=[], clip_names=[], outputs=[], start_scores=[], end_scores=[])
            # start_scores = pool_data(start[None, 0].unsqueeze(2), num_prop=t_scale, num_sample_bin=1)[0, :, 0].cpu().numpy()
            # end_scores = pool_data(end[None, 0].unsqueeze(2), num_prop=t_scale, num_sample_bin=1)[0, :, 0].cpu().numpy()

            anchors = self.anchors1d.last_anchors[confidence_map.device]
            pred_boxes = self.bbox_transform(anchors, box_regression)

            video_duration = int(data.masks[0, ::1, 0].sum())
            cnvrt_output = torch.zeros(video_duration) / 1000

            # pred_boxes = pred_boxes[(anchors[:, :, 1] - anchors[:, :, 0]) > 16]
            # confidence_map = confidence_map[(anchors[:, :, 1] - anchors[:, :, 0]) > 16]

            p_boxes = pred_boxes[confidence_map > self.proposal_threshold]
            c_map = confidence_map[confidence_map > self.proposal_threshold]
            predictions = torch.cat([p_boxes, c_map[:, None]], dim=1)
            # predictions = predictions[torch.logical_or(predictions[:, 0] >= 0, predictions[:, 1] <= video_duration)]
            if len(predictions) != 0:
                # predictions[:, 0] = predictions[:, 0].clip(min=0, max=video_duration)
                # predictions[:, 1] = predictions[:, 1].clip(min=0, max=video_duration)
                # #
                # predictions[:, :2] = predictions[:, :2] / video_duration
                predictions = self.soft_nms(predictions, self.snms_alpha, self.snms_t1, self.snms_t2)
                predictions = predictions[predictions[:, -1] > self.proposal_threshold]
                # predictions[:, :2] = predictions[:, :2] * video_duration

                for prediction in reversed(predictions):
                    prediction = prediction.tolist()
                    cnvrt_output[int(prediction[0]):int(prediction[1])] = prediction[2]
                    # cnvrt_output[int(prediction[0]):int(prediction[1])] = float(prediction[2] > 0.5)

            clip_data.anomalies += data['anomalies'][0].cpu().numpy().tolist()[::1]
            clip_data.outputs += cnvrt_output.cpu().numpy().tolist()
            clip_data.clip_names += data['clip_names'][0][::1]
            # start_scores = start_
            # end_scores = end_
            # clip_data.start_scores += start_scores.detach().cpu().numpy().tolist()
            # clip_data.end_scores += end_scores.detach().cpu().numpy().tolist()
            clip_data.start_scores += [0] * len(clip_data.clip_names)
            clip_data.end_scores += [0] * len(clip_data.clip_names)

            clip_dicts = self.append_overlapped_clips(clip_data)
            stage_clip_dicts = [clip_dicts]

        return loss, report, stage_clip_dicts, loss_dict

    @staticmethod
    def append_overlapped_clips(clip_data):
        clip_dicts = Dict(predictions=dict(), targets=dict(), start_scores=dict(), end_scores=dict())

        for clip_name, prediction, target, s, e in zip(clip_data.clip_names, clip_data.outputs, clip_data.anomalies,
                                                       clip_data.start_scores, clip_data.end_scores):
            if clip_name not in clip_dicts.predictions:
                clip_dicts.predictions[clip_name] = []
            if clip_name not in clip_dicts.targets:
                clip_dicts.targets[clip_name] = []
            if clip_name not in clip_dicts.start_scores:
                clip_dicts.start_scores[clip_name] = []
            if clip_name not in clip_dicts.end_scores:
                clip_dicts.end_scores[clip_name] = []

            clip_dicts.predictions[clip_name].append(float(prediction))
            clip_dicts.targets[clip_name].append(float(target))
            clip_dicts.start_scores[clip_name].append(float(s))
            clip_dicts.end_scores[clip_name].append(float(e))

        return clip_dicts

    @staticmethod
    def move_2_gpu(data, device):
        # obj_features, obj_boxes, masks, anomalies, clip_names, frame_features, match_score_start, \
        # match_score_end, gt_iou_map = data
        obj_features, obj_boxes, masks, anomalies, clip_names, frame_features = data
        gpu_data = Dict()
        if torch.cuda.is_available():
            gpu_data.obj_features = obj_features.float().to(device)
            gpu_data.obj_boxes = obj_boxes.float().to(device)
            gpu_data.anomalies = anomalies.float().to(device)
            gpu_data.masks = masks.float().to(device)
            gpu_data.frame_features = frame_features.float().to(device)
            # gpu_data.match_score_start = match_score_start.float().cuda()
            # gpu_data.match_score_end = match_score_end.float().cuda()
            # gpu_data.gt_iou_map = gt_iou_map.float().cuda()
        gpu_data.clip_names = clip_names
        return gpu_data

    def soft_nms(self, df, alpha, t1, t2):
        '''
        df: proposals generated by network;
        alpha: alpha value of Gaussian decaying function;
        t1, t2: threshold for soft nms.
        '''
        # df = df.sort_values(by="score", ascending=False)
        descending_arg = torch.argsort(df[:, -1], descending=True)
        df = df[descending_arg]

        tstart = df[:, 0].tolist()
        tend = df[:, 1].tolist()
        tscore = df[:, 2].tolist()

        # tstart = list(df.xmin.values[:])
        # tend = list(df.xmax.values[:])
        # tscore = list(df.score.values[:])

        rstart = []
        rend = []
        rscore = []

        # while len(tscore) > 1 and len(rscore) < 101:
        while len(tscore) > 1 and len(rscore) < 101:
            max_index = tscore.index(max(tscore))
            tmp_iou_list = self.ioa_with_anchors(
                np.array(tstart),
                np.array(tend), np.array([tstart[max_index]]), np.array([tend[max_index]]))
            for idx in range(0, len(tscore)):
                if idx != max_index:
                    tmp_iou = tmp_iou_list[idx]
                    tmp_width = tend[max_index] - tstart[max_index]
                    if tmp_iou > t1 + (t2 - t1) * tmp_width:
                        tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) /
                                                           alpha)

            rstart.append(tstart[max_index])
            rend.append(tend[max_index])
            rscore.append(tscore[max_index])

            tstart.pop(max_index)
            tend.pop(max_index)
            tscore.pop(max_index)

        new_df = torch.cat([torch.tensor(rstart)[:, None],
                            torch.tensor(rend)[:, None],
                            torch.tensor(rscore)[:, None]],
                           dim=1).to(df.device)
        return new_df

    # TODO: move to shared utils.py
    @staticmethod
    def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        anchors_min = anchors_min[None, :].repeat(len(box_min), 0)
        anchors_max = anchors_max[None, :].repeat(len(box_max), 0)
        box_min = box_min[:, None].repeat(anchors_min.shape[1], 1)
        box_max = box_max[:, None].repeat(anchors_max.shape[1], 1)
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        # print inter_len,union_len
        jaccard = np.divide(inter_len, union_len)
        return jaccard

    @staticmethod
    def iom_with_anchors(anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        anchors_min = anchors_min[None, :].repeat(len(box_min), 0)
        anchors_max = anchors_max[None, :].repeat(len(box_max), 0)
        box_min = box_min[:, None].repeat(anchors_min.shape[1], 1)
        box_max = box_max[:, None].repeat(anchors_max.shape[1], 1)
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        # len_boxes = box_max - box_min
        # min_len = np.minimum(len_anchors, len_boxes)
        # union_len = len_anchors - inter_len + box_max - box_min
        # print inter_len,union_len
        jaccard = np.divide(inter_len, len_anchors)
        # jaccard = np.divide(inter_len, min_len)
        return np.minimum(jaccard, 1.0)

    @staticmethod
    def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
        # calculate the overlap proportion between the anchor and all bbox for supervise signal,
        # the length of the anchor is 0.01
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def end_of_train_epoch(self, **kwargs):
        pass


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        x_centers_a = (anchors[..., 0] + anchors[..., 1]) / 2
        wa = anchors[..., 1] - anchors[..., 0]

        w = regression[..., 1].exp() * wa

        x_centers = regression[..., 0] * wa + x_centers_a

        xmin = x_centers - w / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, xmax], dim=2)
