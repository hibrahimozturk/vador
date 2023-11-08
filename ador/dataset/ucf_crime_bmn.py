from dataset.ucf_crime import UCFCrimeTemporal, DATASETS
import torch
from utils_.config import Config
import numpy as np
import copy
import re
from metrics import get_labels_start_end_time
import random
import os.path as osp
import math
import time


@DATASETS.register_module(name="ucf_crime_boundary")
class UCFCrimeBoundary(UCFCrimeTemporal):
    def __init__(self, data_cfg, split):
        if hasattr(data_cfg[split], 'clip_slide'):
            self.clip_slide = data_cfg[split].clip_slide
        else:
            self.clip_slide = 4

        UCFCrimeTemporal.__init__(self, data_cfg, split)
        self.frame_feature = getattr(data_cfg, 'frame_feature', False)

        self.t_scale = None
        self.temporal_gap = None
        self.anchor_xmin = None
        self.anchor_xmax = None
        if hasattr(data_cfg[split], 'num_keep_objects'):
            self.num_keep_objects = data_cfg[split].num_keep_objects
        else:
            self.num_keep_objects = None

        # t_scale = int(self.window_size * data_cfg.sampling_rate)
        # self.set_temporal_scale(t_scale)

    def set_temporal_scale(self, tscale):
        self.t_scale = tscale
        self.temporal_gap = 1. / self.t_scale

        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.t_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.t_scale)]

    def prepare_windows(self, annotations, num_sub_windows, maxVideoSize=None, topK=None, regex='.*'):
        video_names = list(annotations["video_clips"].keys())
        r = re.compile(regex)
        video_names = list(filter(r.match, video_names))

        video_names.sort()
        if topK is not None:
            video_names = video_names[:topK]

        clip_lengths = np.array([len(annotations["video_clips"][key][::self.clip_slide]) for key in video_names])
        # print("[Info] Max clip length for dataset is {}".format(np.max(clipLengths)))
        if maxVideoSize is not None:
            filtered_clips = np.where(clip_lengths < maxVideoSize)[0]
            clip_lengths = clip_lengths[filtered_clips]
            video_names = [videoName for i, videoName in enumerate(video_names) if i in filtered_clips]
        num_of_clips = np.sum(clip_lengths)
        # videoWindows = np.ceil(clipLengths / sub_window_size)

        not_included = 0
        for videoIndex, videoName in enumerate(video_names):
            # if self.split == 'train':
            #     start_clip = random.randint(0, self.clip_slide-1)

            # self.clipLists[videoName] = annotations["video_clips"][videoName][start_clip::2]
            # video_clip_list = copy.deepcopy(annotations["video_clips"][videoName][start_clip::2])

            if self.split == 'train':
                video_windows = []
                for start_clip in range(0, self.clip_slide):
                    video_clip_list = copy.deepcopy(annotations["video_clips"][videoName][start_clip::self.clip_slide])
                    # if len(video_clip_list) < self.window_size:
                    #     clip_ids = np.linspace(0, len(video_clip_list) - 1, min(3 * len(video_clip_list), self.window_size), dtype=int)
                    #     video_clip_list = [video_clip_list[c_id] for c_id in clip_ids]
                    temporal_gt = [annotations['all_clips'][clip_name].anomaly for clip_name in video_clip_list]
                    labels, starts, ends = get_labels_start_end_time(temporal_gt)
                    if len(video_clip_list) < self.window_size:
                        pad_length = self.window_size - len(video_clip_list)
                        video_clip_list += [""] * pad_length
                        temporal_gt += [-1]*pad_length
                    if len(starts) != 0:
                        for action_order in range(len(starts)):
                            pre_search = [ends[action_order - 1] if action_order != 0 else 0, starts[action_order]]
                            post_search = [ends[action_order],
                                           starts[action_order + 1] if action_order != len(starts) - 1 else len(temporal_gt)]
                            # if pre_search[1] - pre_search[0] >= self.window_size:
                            #     start = random.randint(pre_search[0], pre_search[1]-self.window_size)
                            #     window = video_clip_list[start:start + self.window_size]
                            #     assert len(window) == self.window_size, "window size does not match"
                            #     w_anomalies = np.array([self.clips[clip]['anomaly'] for clip in window if clip != ""])
                            #     self.windows.append(window)
                            # if post_search[1] - post_search[0] >= self.window_size:
                            #     start = random.randint(post_search[0], post_search[1]-self.window_size)
                            #     window = video_clip_list[start:start + self.window_size]
                            #     assert len(window) == self.window_size, "window size does not match"
                            #     w_anomalies = np.array([self.clips[clip]['anomaly'] for clip in window if clip != ""])
                            #     self.windows.append(window)
                            post_search[1] = min(post_search[1], starts[action_order] + self.window_size)
                            pre_search[0] = max(pre_search[0], ends[action_order] - self.window_size)
                            i = 1
                            while post_search[1] - pre_search[0] < self.window_size and action_order + i < len(labels):
                                post_search = [ends[action_order + i],
                                               starts[action_order + 1 + i] if action_order + i != len(starts) - 1 else len(
                                                   temporal_gt)]
                                post_search[1] = min(post_search[1], starts[action_order] + self.window_size)

                                i += 1

                            max_area = post_search[1] - pre_search[0]
                            min_area = post_search[0] - pre_search[1]
                            if max_area >= self.window_size >= min_area:
                                pre_search[1] = post_search[1] - self.window_size
                                start = random.choice(range(pre_search[0], pre_search[1]+1))
                                window = video_clip_list[start:start + self.window_size]
                                assert len(window) == self.window_size, "window size does not match"
                                w_anomalies = np.array([self.clips[clip]['anomaly'] for clip in window if clip != ""])

                                self.windows.append(window)
                                contains_anomaly = (np.array(temporal_gt[start:start + self.window_size]) == 1).any()
                                self.windows_contains_anomaly.append(bool(contains_anomaly))
                                for possible_start in range(pre_search[0], pre_search[1] + 1):
                                    window = video_clip_list[possible_start:possible_start + self.window_size]
                                    assert len(window) == self.window_size, "window size does not match"
                                    video_windows.append(window)
                            else:
                                not_included += 1
                    else:
                        len_ratio = len(video_clip_list) / self.window_size
                        start_map = np.zeros(len(video_clip_list))
                        start_map[0: len(video_clip_list) - self.window_size + 1] = 1

                        for possible_start in (start_map == 1).nonzero()[0]:
                            window = video_clip_list[possible_start:possible_start + self.window_size]
                            video_windows.append(window)

                        for _ in range(10):
                            start = np.random.choice(np.nonzero(start_map != 0)[0])
                            # start = random.choice(range(0, len(video_clip_list) - self.window_size + 1))
                            window = video_clip_list[start:start + self.window_size]
                            self.windows.append(window)
                            self.windows_contains_anomaly.append(False)
                            start_map[max(start - int(self.window_size / 2), 0):
                                      min(len(video_clip_list) - 1, start + int(self.window_size / 2))] = 0
                            if not start_map.any():
                                break

                if len(video_windows) != 0:
                    self.video_windows.append(video_windows)
            else:
                start_clip = 0
                self.clipLists[videoName] = annotations["video_clips"][videoName][start_clip::self.clip_slide]
                # self.clipLists[videoName] = annotations["video_clips"][videoName][0::1]
                # for start_clip in range(0, self.clip_slide):
                # for start_clip in range(0, 1):
                video_clip_list = copy.deepcopy(annotations["video_clips"][videoName][start_clip::self.clip_slide])
                window_size = max(len(video_clip_list), self.window_size*1)  # TODO remove after test
                ideal_window_size = int(np.ceil(window_size / 32) * 32)
                # clip_ids = np.linspace(0, len(video_clip_list) - 1, min(3 * len(video_clip_list),  ideal_window_size), dtype=int)
                clip_ids = np.arange(0, len(video_clip_list), 1)
                video_clip_list = [video_clip_list[c_id] for c_id in clip_ids]
                pad_list = [""] * int(ideal_window_size - len(video_clip_list))
                self.windows.append(video_clip_list + pad_list)
                self.video_windows.append([video_clip_list + pad_list])

        return num_of_clips

    def __len__(self):
        return len(self.video_windows)

    @staticmethod
    def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        # print inter_len,union_len
        jaccard = np.divide(inter_len, union_len)
        return jaccard

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

    def __getitem__(self, item):
        # print(self.t_scale)
        windowClipNames = random.choice(self.video_windows[item])
        num_objs = self.num_keep_objects if self.num_keep_objects is not None else 18
        obj_features = np.zeros((len(windowClipNames), num_objs, self.featureSize))
        obj_boxes = np.zeros((len(windowClipNames), num_objs, 4))
        frame_features = np.zeros((len(windowClipNames), 1, 1024, 7, 7))
        # frame_boxes = np.zeros((len(windowClipNames), 4))
        # frame_features = np.zeros((self.window_size, 1, self.featureSize))
        # anomalies = []
        # categories = []
        masks = np.zeros((len(windowClipNames), 1))
        # categories = np.zeros(len(windowClipNames), 1)

        # current_group = None
        # current_group_feat = None

        group_size = 10
        anomalies = [self.clips[clipName]['anomaly'] for clipName in windowClipNames if clipName != ""]
        masks[:len(anomalies)] = 1
        anomalies = anomalies + [0] * (len(windowClipNames) - len(anomalies))
        clip_orders = [int(clipName.rsplit('_', 1)[-1]) for clipName in windowClipNames if clipName != ""]
        clip_orders = np.array(clip_orders)
        group_ids = np.floor(clip_orders / (5 * group_size))
        orders_in_group = clip_orders / 5 - group_ids * group_size
        groups = np.unique(group_ids)

        data_path = random.choice(self.clips[windowClipNames[0]]["data_paths"])
        # print('num of groups: {}'.format(len(groups)))
        for group_id in groups:
            group_name = '{:05}-{:05}'.format(int(group_id) * 5 * group_size, (int(group_id) + 1) * 5 * group_size)
            video_name = self.clips[windowClipNames[0]]['video_name']
            group_feats = np.load(osp.join(data_path + '_compressed', video_name, group_name + '.npz'), mmap_mode='r')
            mask = group_ids == group_id
            obj_features_tmp = group_feats['obj_features'][orders_in_group[mask].astype('int')]
            obj_boxes_tmp = group_feats['boxes'][orders_in_group[mask].astype('int')]
            if self.num_keep_objects is not None:
                if self.split == 'train':
                    selected_objects = np.random.choice(obj_features_tmp.shape[1],
                                                        size=(obj_features_tmp.shape[0], self.num_keep_objects))
                    selected_rows = torch.arange(len(obj_features_tmp))[:, None].repeat(1, self.num_keep_objects)
                    obj_features_tmp = obj_features_tmp[selected_rows, selected_objects, :]
                    obj_boxes_tmp = obj_boxes_tmp[selected_rows, selected_objects, :]
                else:
                    obj_features_tmp = obj_features_tmp[:, :self.num_keep_objects, :]
                    obj_boxes_tmp = obj_boxes_tmp[:, :self.num_keep_objects, :]

            obj_features[:len(mask)][mask] = obj_features_tmp
            obj_boxes[:len(mask)][mask] = obj_boxes_tmp
            frame_features[:len(mask)][mask] = group_feats['frame_features'][orders_in_group[mask].astype('int')][:, None]
            # time.sleep(0.05)

        # for index, clipName in enumerate(windowClipNames):
        #     if clipName != "":
        #         video_name = self.clips[clipName]["video_name"]
        #         frame_order = self.clips[clipName]["frame_order"]
        #         # data_path = random.choice(self.clips[clipName]["data_paths"])
        #
        #         group_id = math.floor(int(frame_order) / 500)
        #         if group_id != current_group:
        #             group_name = '{:05}-{:05}'.format(group_id * 500, (group_id + 1)*500)
        #             current_group_feat = np.load(osp.join(data_path + '_compressed', video_name, group_name + '.npz'))
        #             current_group = group_id
        #
        #         order_in_group = int(int(frame_order)/5 - 100*group_id)
        #         obj_box = current_group_feat['boxes'][order_in_group]
        #         obj_feature = current_group_feat['obj_features'][order_in_group]
        #         # obj_feature = np.load(osp.join(data_path, video_name + '_features', frame_order + '.npz'))['arr_0']
        #         # obj_box = np.load(osp.join(data_path, video_name + '_box', frame_order + '.npz'))['arr_0']
        #         keep_ids = np.arange(len(obj_feature))
        #         if self.num_keep_objects is not None:
        #             if self.split == 'train':
        #                 np.random.shuffle(keep_ids)
        #             keep_ids = keep_ids[:self.num_keep_objects]
        #
        #         frame_feature = current_group_feat['frame_features'][order_in_group]
        #         # frame_feature = np.load(osp.join(data_path, video_name + '_i3d', frame_order + '.npz'))['arr_0']
        #         obj_feature = obj_feature[keep_ids]
        #         obj_box = obj_box[keep_ids]
        #         frame_box = np.zeros(4)
        #         mask = 1
        #         anomalies.append(self.clips[clipName]["anomaly"])
        #         categories.append(self.clips[clipName]["category"])
        #     else:
        #
        #         frame_feature = np.zeros((1, 1024, 7, 7))
        #         frame_box = np.zeros(4)
        #         obj_feature = np.zeros((18, self.featureSize)) + self.mask_value
        #         obj_box = np.zeros((18, 4))
        #         if self.num_keep_objects is not None:
        #             obj_feature = obj_feature[:self.num_keep_objects]
        #             obj_box = obj_box[:self.num_keep_objects]
        #         mask = 0
        #         # anomalies.append(-1)
        #         anomalies.append(0)
        #         categories.append(-1)
        #     obj_features[index, :] = obj_feature
        #     obj_boxes[index, :] = obj_box
        #     masks[index, :] = mask
        #     frame_features[index, :, :] = frame_feature
        #     frame_boxes[index] = frame_box

        obj_features = torch.from_numpy(obj_features)
        obj_boxes = torch.from_numpy(obj_boxes)
        obj_features = obj_features[:, :18, :]
        obj_boxes = obj_boxes[:, :18, :]
        anomalies = torch.from_numpy(np.array(anomalies))
        frame_features = torch.from_numpy(frame_features)
        # frame_boxes = torch.from_numpy(frame_boxes)

        # abnormal video normal clips classified as normal (15)
        # categoriesMerged = np.ones_like(anomalies) * 15
        # categoriesMerged[anomalies == 1] = categories[anomalies == 1]

        # categories = np.array(categories)
        # categories = torch.from_numpy(categories)
        masks = torch.from_numpy(np.array(masks))
        return obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features

        # return obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features, match_score_start, \
        #        match_score_end, gt_iou_map

    @staticmethod
    def get_feature(path):
        pass

    @staticmethod
    def collate_fn(data):
        # obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features, match_score_start, \
        # match_score_end, gt_iou_map = zip(*data)

        obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features = zip(*data)

        obj_features = torch.stack(obj_features, 0)
        obj_boxes = torch.stack(obj_boxes, 0)
        anomalies = torch.stack(anomalies, 0)
        masks = torch.stack(masks, 0)
        # categories = torch.stack(categories, 0)
        frame_features = torch.stack(frame_features, 0)
        # match_score_start = torch.stack(match_score_start, 0)
        # match_score_end = torch.stack(match_score_end, 0)
        # gt_iou_map = torch.stack(gt_iou_map, 0)
        # frame_boxes = torch.stack(frame_boxes, 0)

        # return obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features, match_score_start, \
        #        match_score_end, gt_iou_map

        return obj_features, obj_boxes, masks, anomalies, windowClipNames, frame_features


if __name__ == "__main__":
    cfg = Config.fromfile('../../experiments/ADOR/configs/dataset_cfg.py')
    for split in ['train', 'val', 'test']:
        for part in ['normal', 'abnormal']:
            getattr(getattr(cfg.dataset, split), part).clip_features = "../" + getattr(getattr(cfg.dataset, split),
                                                                                       part).clip_features
            getattr(getattr(cfg.dataset, split), part).annotations = "../" + getattr(getattr(cfg.dataset, split),
                                                                                     part).annotations

    t = UCFCrimeObjects(cfg.dataset, 'train')
    a = t[0]
