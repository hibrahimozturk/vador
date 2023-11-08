import copy

from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import os
from tqdm import tqdm
import torch
from addict import Dict
from utils_.registry import Registry
import re
import time

import logging

logger = logging.getLogger("violence")

DATASETS = Registry("datasets")


@DATASETS.register_module(name="ucf_crime_dataset")
class UCFCrimeDataset(Dataset):
    def __init__(self, data_cfg, split):

        split_cfg = getattr(data_cfg, split)
        start = time.time()
        with open(split_cfg.abnormal.annotations) as fp:
            self.abnormal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.abnormal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))
        logger.info('reading abnormal json takes {} second'.format(time.time() - start))
        start = time.time()
        with open(split_cfg.normal.annotations) as fp:
            self.normal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.normal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))
        logger.info('reading normal json takes {} second'.format(time.time() - start))

        self.clips = []
        self.clipLists = {}

        for clipName, value in self.abnormal.annotations["all_clips"].items():
            self.clips.append({"path": os.path.join(self.abnormal.clip_features, clipName),
                               "anomaly": value["anomaly"],
                               "category": value["category"],
                               "category_name": value["category_name"],
                               "clip_name": clipName})

        for clipName, value in self.normal.annotations["all_clips"].items():
            self.clips.append({"path": os.path.join(self.normal.clip_features, clipName),
                               "anomaly": value["anomaly"],
                               "category": value["category"],
                               "category_name": value["category_name"],
                               "clip_name": clipName})

        for videoIndex, videoName in enumerate(self.abnormal.annotations["video_clips"]):
            self.clipLists[videoName] = self.abnormal.annotations["video_clips"][videoName]

        for videoIndex, videoName in enumerate(self.normal.annotations["video_clips"]):
            self.clipLists[videoName] = self.normal.annotations["video_clips"][videoName]

        # logger.info("Dataset has been constructed")
        # logger.info("# abnormal clips: {}".format(len(self.abnormal.annotations["abnormal_clips"])))
        # logger.info("# normal clips: {}".format(len(self.abnormal.annotations["normal_clips"])))

    def __len__(self):
        return len(self.clips)

    def __getVideoClips__(self):
        return self.clipLists

    @staticmethod
    def get_feature(path):
        feature = np.load(path + ".npy")
        return feature

    def __getitem__(self, idx):
        annotation = self.clips[idx]
        feature = np.load(os.path.join(annotation["path"]))

        return feature, 1, annotation["anomaly"], annotation["category"], \
               annotation["category_name"], annotation["clip_name"]

    @staticmethod
    def collate_fn(data):
        inputData, masks, anomalies, categories, categoryNames, windowClipNames = zip(*data)

        inputData = torch.stack(inputData, 0)
        anomalies = torch.stack(anomalies, 0)
        masks = torch.stack(masks, 0)
        categories = torch.stack(categories, 0)

        return inputData, masks, anomalies, categories, categoryNames, windowClipNames


@DATASETS.register_module(name="ucf_crime_temporal")
class UCFCrimeTemporal(Dataset):
    def __init__(self, data_cfg, split):

        self.split = split
        split_cfg = getattr(data_cfg, split)
        start = time.time()
        with open(split_cfg.abnormal.annotations, 'rb') as fp:
            self.abnormal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.abnormal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None,
                regex=split_cfg.abnormal.regex if hasattr(split_cfg.abnormal, 'regex') else '.*'
            ))
        logger.info('reading abnormal json takes {} second'.format(time.time() - start))
        start = time.time()
        with open(split_cfg.normal.annotations) as fp:
            self.normal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.normal.clip_features,
                top_k=split_cfg.normal.top_k if hasattr(split_cfg.normal, "top_k") else None,
                regex=split_cfg.normal.regex if hasattr(split_cfg.normal, 'regex') else '.*'

            ))
        logger.info('reading normal json takes {} second'.format(time.time() - start))

        self.clips = {}
        for part in [self.abnormal, self.normal]:
            for clip_name, value in part.annotations.all_clips.items():
                video_name, frame_order = clip_name.rsplit('_', 1)
                self.clips[clip_name] = {"path": os.path.join(part.clip_features[0], clip_name),
                                         "anomaly": value["anomaly"],
                                         "category": value["category"],
                                         "category_name": value["category_name"],
                                         "video_name": video_name,
                                         "frame_order": frame_order,
                                         "data_paths": part.clip_features}

        self.window_size = data_cfg.window_size
        self.featureSize = data_cfg.feature_size
        # self.sub_window_size = int(self.window_size / data_cfg.sub_windows)
        # self.num_sub_windows = data_cfg.sub_windows
        self.mask_value = data_cfg.mask_value
        self.clipLists = {}
        self.windows = []
        self.windows_contains_anomaly = []
        self.video_windows = []

        total_windows = 0
        for part_name, part in dict(normal=self.normal, abnormal=self.abnormal).items():
            num_clips = self.prepare_windows(part.annotations, split_cfg.sub_windows,
                                             maxVideoSize=data_cfg[split].max_video_len,
                                             topK=part.top_k, regex=part.regex)
            part.num_clips = num_clips
            logger.info("num of {} windows in {}: {}".format(part_name, self.split, len(self.windows) - total_windows))
            total_windows += len(self.windows)
            # if hasattr(part, "top_k"):
            #     logger.info("first {} {} videos have been included".format(part.top_k, part_name))

        # Abnormal clips in only abnormal videos
        # Normal clips are in normal and abnormal videos
        # logger.info("# abnormal clips: {}".format(self.abnormal.num_clips))
        # logger.info("# normal clips: {}".format(self.abnormal.num_clips + self.normal.num_clips))
        # logger.info("dataset has been constructed")

    def prepare_windows(self, annotations, num_sub_windows, maxVideoSize=None, topK=None, regex='.*'):
        sub_window_size = int(self.window_size / num_sub_windows)
        videoNames = list(annotations["video_clips"].keys())
        r = re.compile(regex)
        videoNames = list(filter(r.match, videoNames))

        videoNames.sort()
        if topK is not None:
            videoNames = videoNames[:topK]
        clipLengths = np.array([len(annotations["video_clips"][key]) for key in videoNames])
        # print("[Info] Max clip length for dataset is {}".format(np.max(clipLengths)))
        if maxVideoSize is not None:
            filteredClips = np.where(clipLengths < maxVideoSize)[0]
            clipLengths = clipLengths[filteredClips]
            videoNames = [videoName for i, videoName in enumerate(videoNames) if i in filteredClips]
        numOfClips = np.sum(clipLengths)
        videoWindows = np.ceil(clipLengths / sub_window_size)
        for videoIndex, videoName in enumerate(videoNames):
            self.clipLists[videoName] = annotations["video_clips"][videoName]
            videoClipList = copy.deepcopy(annotations["video_clips"][videoName])
            for clipIndex in range(len(videoClipList),
                                   max(int(videoWindows[videoIndex] * sub_window_size), self.window_size)):
                videoClipList.append("")

            for windowIndex in range(max(int(videoWindows[videoIndex]) - (num_sub_windows - 1), 1)):
                start = windowIndex * sub_window_size
                window = videoClipList[start:start + self.window_size]
                assert len(window) == self.window_size, "window size does not match"
                self.windows.append(window)
        return numOfClips

    def __len__(self):
        return len(self.windows)

    def __getVideoClips__(self):
        return self.clipLists

    def __get_video_info__(self):
        return self.normal.annotations['video_info'] | self.abnormal.annotations['video_info']

    @staticmethod
    def get_feature(path):
        feature = np.load(path + ".npy")
        return feature

    def __getitem__(self, item):
        windowClipNames = self.windows[item]
        inputData = np.zeros((self.window_size, self.featureSize))
        anomalies = []
        categories = []
        categoryNames = []
        masks = np.zeros((self.window_size, 1))
        for index, clipName in enumerate(windowClipNames):
            if clipName != "":
                # clip = np.load(self.clips[clipName]["path"])
                clip = self.get_feature(self.clips[clipName]["path"])
                mask = 1
                # annotation = self.abnormalAnnotations["allClips"][clipName]
                anomalies.append(self.clips[clipName]["anomaly"])
                categories.append(self.clips[clipName]["category"])
                categoryNames.append((self.clips[clipName]["category_name"]))
            else:
                clip = np.zeros(self.featureSize) + self.mask_value
                mask = 0
                anomalies.append(-1)
                categories.append(-1)
                categoryNames.append("")
            inputData[index, :] = clip
            masks[index, :] = mask
        inputData = torch.from_numpy(inputData)
        anomalies = torch.from_numpy(np.array(anomalies))

        # abnormal video normal clips classified as normal (15)
        categoriesMerged = np.ones_like(anomalies) * 15
        categories = np.array(categories)
        categoriesMerged[anomalies == 1] = categories[anomalies == 1]

        categories = torch.from_numpy(np.array(categories))
        masks = torch.from_numpy(np.array(masks))
        return inputData, masks, anomalies, categories, categoryNames, windowClipNames

    @staticmethod
    def collate_fn(data):
        inputData, masks, anomalies, categories, categoryNames, windowClipNames = zip(*data)

        inputData = torch.stack(inputData, 0)
        anomalies = torch.stack(anomalies, 0)
        masks = torch.stack(masks, 0)
        categories = torch.stack(categories, 0)

        return inputData, masks, anomalies, categories, categoryNames, windowClipNames


if __name__ == "__main__":
    valSet = UCFCrimeTemporal("../data/i3d_features/abnormal/train",
                              "../data/i3d_features/abnormal/TrainLabels.json",
                              "../data/i3d_features/normal/train",
                              "../data/i3d_features/normal/TrainLabels.json",
                              normalTopK=100)
    dataloader = DataLoader(valSet, batch_size=16, shuffle=True, num_workers=20,
                            collate_fn=UCFCrimeTemporal.collate_fn)
    for i, batch in enumerate(tqdm(dataloader)):
        i
