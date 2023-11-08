import random

from handlers.handler import HANDLERS
from handlers.temporal_anomaly import TemporalAnomalyDetectionHandler

from models.TALNet.TALNet import TALNet
import numpy as np
from scipy import sparse

from addict import Dict
import torch
import torch.nn as nn
import itertools
import math
import os.path as osp
import metrics
from utils_.utils import visualize_temporal_action_bmn
from utils_.anchors1d import Anchors1D
import pickle

from tqdm import autonotebook


@HANDLERS.register_module(name="bmn_v4_")
class BMNV4Handler(TemporalAnomalyDetectionHandler):
    """
    Box prediction is made by regression head in BMN, instead of start, end predictions
    """

    def __init__(self, cfg):
        TemporalAnomalyDetectionHandler.__init__(self, cfg)

        self.proposal_threshold = cfg.evaluation.proposal_threshold
        self.anchors1d = Anchors1D(anchor_scale=cfg.obj_model.anchor_scale,
                                   pyramid_levels=list(range(cfg.obj_model.fpn_levels)))
        # self.tscale = cfg.model.TALNet.temporal_length
        # self.window_size = cfg.model.TALNet.window_size

        self.prop_boundary_ratio = cfg.obj_model.prop_boundary_ratio
        self.num_sample = cfg.obj_model.num_sample
        self.num_sample_perbin = cfg.obj_model.num_sample_perbin
        # self.sample_mask = self._get_interp1d_mask(self.tscale)
        self.sample_mask = None
        # self.feat_dim = cfg.model.TALNet.feat_dim

        self.snms_alpha = cfg.evaluation.soft_nms.alpha
        self.snms_t1 = cfg.evaluation.soft_nms.low_threshold
        self.snms_t2 = cfg.evaluation.soft_nms.high_threshold

        self.temporal_lengths = cfg.obj_model.temporal_lengths

        self.stage_keys = ['relation_transformer']
        # self.bm_mask = self.get_mask(self.tscale)
        self.bm_mask = None

        # if hasattr(cfg.model, 'TALNet'):
        #     self.stage_keys += ['mstcn.' + key for key, _ in self.model.mstcn._modules.items() if 'stage_' in key]

        # if hasattr(cfg.model, 'active_stages'):
        #     self.active_stages = torch.tensor(cfg.model.active_stages)
        # else:
        #     total_stages = cfg.model.mstcn.num_stages + 1
        #     self.active_stages = torch.tensor([-1] * total_stages)

        self.temp_ratio = []
        self.bbox_transform = BBoxTransform()

    def get_model(self, model_cfg):
        model = TALNet(feat_dim=model_cfg.feat_dim, fpn_levels=model_cfg.fpn_levels)
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
        anchors = self.anchors1d(anomalies)

        anchor_widths = anchors[0, :, 1] - anchors[0, :, 0]
        anchor_ctr_x = anchors[0, :, 0] + 0.5 * anchor_widths

        temporal_gap = 1. / tscale
        anchor_xmin = [temporal_gap * (i - 0.5) for i in range(tscale)]
        anchor_xmax = [temporal_gap * (i + 0.5) for i in range(tscale)]

        for anomaly_map in anomalies:
            gt_bbox = []
            bboxes = []
            labels, starts, ends = metrics.get_labels_start_end_time(anomaly_map)
            for start, end in zip(starts, ends):
                gt_bbox.append([start / len(anomaly_map), end / len(anomaly_map)])
                bboxes.append([start, end])

            bboxes = np.array(bboxes)
            gt_bbox = np.array(gt_bbox)

            if len(gt_bbox) != 0:
                gt_xmins = gt_bbox[:, 0]
                gt_xmaxs = gt_bbox[:, 1]
                gt_lens = gt_xmaxs - gt_xmins
                gt_len_small = 3 * temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
                gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
                gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

                gt_iou_map = self.iou_with_anchors(anchors[0, :, 0].cpu().numpy(), anchors[0, :, 1].cpu().numpy(),
                                                   bboxes[:, 0], bboxes[:, 1])

                gt_iom_map = self.iom_with_anchors(anchors[0, :, 0].cpu().numpy(), anchors[0, :, 1].cpu().numpy(),
                                                   bboxes[:, 0], bboxes[:, 1])

                gt_iou_map = torch.Tensor(gt_iou_map)
                gt_iom_map = torch.Tensor(gt_iom_map)
                iou_max, iou_argmax = gt_iou_map.max(dim=0)
                iom_max, _ = gt_iom_map.max(dim=0)
                selected_boxes = bboxes[iou_argmax, :]
                gt_iom_map = iom_max >= 0.5
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
                gt_iom_map = torch.zeros(anchors.shape[1])
                gt_iou_map = torch.zeros(anchors.shape[1])
                targets = torch.zeros((anchors.shape[1], 2)).to(anomalies.device)
                match_score_start = torch.zeros(tscale)
                match_score_end = torch.zeros(tscale)

            iom_maps.append(gt_iom_map)
            iou_maps.append(gt_iou_map)
            regression_targets.append(targets)

        # start_scores = torch.stack(start_scores).to(anomalies.device)
        # end_scores = torch.stack(end_scores).to(anomalies.device)
        regression_targets = torch.stack(regression_targets)
        iom_maps = torch.stack(iom_maps).to(anomalies.device)
        iou_maps = torch.stack(iou_maps).to(anomalies.device)
        return iom_maps.float(), iou_maps.float(), regression_targets

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

    # @staticmethod
    # def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    #     """Compute jaccard score between a box and the anchors.
    #     """
    #     len_anchors = anchors_max - anchors_min
    #     int_xmin = np.maximum(anchors_min, box_min)
    #     int_xmax = np.minimum(anchors_max, box_max)
    #     inter_len = np.maximum(int_xmax - int_xmin, 0.)
    #     union_len = len_anchors - inter_len + box_max - box_min
    #     # print inter_len,union_len
    #     jaccard = np.divide(inter_len, union_len)
    #     return jaccard

    # @staticmethod
    # def iom_with_anchors(anchors_min, anchors_max, box_min, box_max):
    #     """
    #     Intersection over Minimum
    #     Compute jaccard score between a box and the anchors.
    #     """
    #     len_anchors = anchors_max - anchors_min
    #     int_xmin = np.maximum(anchors_min, box_min)
    #     int_xmax = np.minimum(anchors_max, box_max)
    #     inter_len = np.maximum(int_xmax - int_xmin, 0.)
    #     len_boxes = box_max - box_min
    #     min_len = np.minimum(len_anchors, len_boxes)
    #     # union_len = len_anchors - inter_len + box_max - box_min
    #     # print inter_len,union_len
    #     jaccard = np.minimum(np.divide(inter_len, min_len), 1.0)
    #     return jaccard

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
            clips = Dict(predictions=[], targets=[], start_scores=[], end_scores=[])
            for clip_name in clip_list:
                # # TODO: for TALNet, it will be deleted !!!!!!!!!!!!
                # if clip_name not in predictions:
                #     predictions[clip_name] = [0]
                #
                # if clip_name not in targets:
                #     targets[clip_name] = [0]

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
            visualize_temporal_action_bmn(clips.predictions, clips.targets, clips.start_scores, clips.end_scores,
                                          save_path, video_name)
        with open(osp.join(pickels_dir, 'clips{}.pickle'.format(file_ext.split('.')[0])), 'wb') as fp:
            pickle.dump(output_json, fp)

    def model_forward(self, data, evaluate=False, **kwargs):
        data = self.move_2_gpu(data)

        report = dict()
        stage_clip_dicts = None
        loss_dict = {"{}_loss".format(loss_type): [] for loss_type, loss_cfg in self.losses.items()}

        # if self.sample_mask.shape[0] != len(data['anomalies'][0]):
        #     self.sample_mask = self._get_interp1d_mask(len(data['anomalies'][0]))
        #     self.bm_mask = self.get_mask(len(data['anomalies'][0]))

        # if self.mode == 'train':
        #     t_scale = random.choice(self.temporal_lengths.train)
        # else:
        #     t_scale = self.temporal_lengths.eval[0]

        # self.sample_mask = self._get_interp1d_mask(t_scale)
        # self.bm_mask = self.get_mask(t_scale)

        iom_maps, iou_maps, regression_targets = self.prepare_target_maps(data.anomalies, data.masks,
                                                                len(data.anomalies[0]))

        # iou_maps, _, _ = self.prepare_target_maps(data.anomalies, t_scale)
        # _, start_scores, end_scores = self.prepare_target_maps(data.anomalies, len(data.anomalies[0]))
        # start_scores = start_scores[:, data.masks[0, :, 0] != 0]
        # end_scores = end_scores[:, data.masks[0, :, 0] != 0]

        confidence_map, box_regression = self.model(data.frame_features[:, :, 0].mean(-1).mean(-1).transpose(1, 2))

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

            video_duration = int(data.masks[0, :, 0].sum())
            cnvrt_output = torch.zeros(video_duration)

            # pred_boxes = pred_boxes[(anchors[:, :, 1] - anchors[:, :, 0]) > 16]
            # confidence_map = confidence_map[(anchors[:, :, 1] - anchors[:, :, 0]) > 16]

            p_boxes = pred_boxes[confidence_map > self.proposal_threshold]
            c_map = confidence_map[confidence_map > self.proposal_threshold]
            predictions = torch.cat([p_boxes, c_map[:, None]], dim=1)

            if len(predictions) != 0:

                predictions = self.soft_nms(predictions, self.snms_alpha, self.snms_t1, self.snms_t2)
                predictions = predictions[predictions[:, -1] > self.proposal_threshold]

                for prediction in reversed(predictions):
                    prediction = prediction.tolist()
                    cnvrt_output[int(prediction[0]):int(prediction[1])] = prediction[2]
                    # cnvrt_output[int(prediction[0]):int(prediction[1])] = float(prediction[2] > 0.5)

            clip_data.anomalies += data['anomalies'][0].cpu().numpy().tolist()
            clip_data.outputs += cnvrt_output.cpu().numpy().tolist()
            clip_data.clip_names += data['clip_names'][0]
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

    def _get_interp1d_mask(self, tscale):
        # generate sample mask for each point in Boundary-Matching Map
        # mask_mat = []
        cache_path = osp.join('.cache/bmn_masks',
                              't{}_t{}_n{}_b{}'.format(tscale, tscale, self.num_sample,
                                                       self.num_sample_perbin))

        if osp.exists(cache_path + '.npz'):
            mask_mat = sparse.load_npz(cache_path + '.npz').todense()
        else:
            # TODO: mask creation with sparse matrices
            mask_mat = np.zeros([tscale, self.num_sample, tscale, tscale], dtype='float32')
            # mask_mat = sparse.csr_matrix([tscale, self.num_sample, tscale, tscale], dtype='float32')
            for end_index in autonotebook.tqdm(range(tscale)):
                # mask_mat_vector = []
                mask_mat_vector = np.zeros([tscale, self.num_sample, tscale], dtype='float32')
                # mask_mat_vector = sparse.csr_matrix([tscale, self.num_sample, tscale], dtype='float32')
                for start_index in range(tscale):
                    if start_index <= end_index:
                        p_xmin = start_index
                        p_xmax = end_index + 1
                        center_len = float(p_xmax - p_xmin) + 1
                        sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                        sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                        p_mask = self._get_interp1d_bin_mask(
                            sample_xmin, sample_xmax, tscale, self.num_sample,
                            self.num_sample_perbin)
                        mask_mat_vector[:, :, start_index] = p_mask
                    else:
                        # p_mask = np.zeros([self.tscale, self.num_sample])
                        break
                    # mask_mat_vector.append(p_mask)
                # mask_mat_vector = np.stack(mask_mat_vector, axis=2)
                # mask_mat.append(mask_mat_vector)
                mask_mat[:, :, :, end_index] = mask_mat_vector
            # mask_mat = np.stack(mask_mat, axis=3)
            mask_mat = mask_mat.astype(np.float32)
            sparse.save_npz(cache_path, sparse.csr_matrix(mask_mat.reshape(tscale, -1)))

        sample_mask = nn.Parameter(torch.tensor(mask_mat).view(tscale, -1).float(), requires_grad=False)
        if torch.cuda.is_available():
            sample_mask = sample_mask.cuda()
        return sample_mask

    @staticmethod
    def _get_interp1d_bin_mask(seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = np.zeros((tscale, num_sample), dtype='float32')
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale]).astype('float32')
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if (tscale - 1) >= int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if (tscale - 1) >= int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            # p_mask.append(bin_vector)
            p_mask[:, idx] = bin_vector
        # p_mask = np.stack(p_mask, axis=1)
        return p_mask

    @staticmethod
    def get_mask(tscale):
        mask = np.zeros([tscale, tscale], np.float32)
        for i in range(tscale):
            for j in range(i, tscale):
                mask[i, j] = 1

        mask = torch.Tensor(mask)
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    @staticmethod
    def move_2_gpu(data):
        # obj_features, obj_boxes, masks, anomalies, clip_names, frame_features, match_score_start, \
        # match_score_end, gt_iou_map = data
        obj_features, obj_boxes, masks, anomalies, clip_names, frame_features = data
        gpu_data = Dict()
        if torch.cuda.is_available():
            gpu_data.obj_features = obj_features.float().cuda()
            gpu_data.obj_boxes = obj_boxes.float().cuda()
            gpu_data.anomalies = anomalies.float().cuda()
            gpu_data.masks = masks.float().cuda()
            gpu_data.frame_features = frame_features.float().cuda()
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

        while len(tscore) > 1 and len(rscore) < 101:
            max_index = tscore.index(max(tscore))
            tmp_iou_list = self.iom_with_anchors(
                np.array(tstart),
                np.array(tend), np.array([tstart[max_index]]), np.array([tend[max_index]]))
            for idx in range(0, len(tscore)):
                if idx != max_index:
                    tmp_iou = tmp_iou_list[0, idx]
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

    def end_of_train_epoch(self, **kwargs):
        # In order to create random windows in each epoch
        # self.train_loader = self.get_dataloaders(self.cfg.dataset, splits=['train'], dist_cfg=self.cfg.dist_dict)
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
