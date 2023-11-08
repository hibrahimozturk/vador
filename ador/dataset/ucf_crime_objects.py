from dataset.ucf_crime import UCFCrimeTemporal, DATASETS
import torch
from utils_.config import Config
import numpy as np


@DATASETS.register_module(name="ucf_crime_objects")
class UCFCrimeObjects(UCFCrimeTemporal):
    def __init__(self, data_cfg, split):
        UCFCrimeTemporal.__init__(self, data_cfg, split)
        self.frame_feature = getattr(data_cfg, 'frame_feature', False)

    def __getitem__(self, item):
        windowClipNames = self.windows[item]
        obj_features = np.zeros((self.window_size, 36, self.featureSize))
        obj_boxes = np.zeros((self.window_size, 36, 4))
        frame_features = np.zeros((self.window_size, 1, 1024, 7, 7))
        frame_boxes = np.zeros((self.window_size, 4))
        # frame_features = np.zeros((self.window_size, 1, self.featureSize))
        anomalies = []
        categories = []
        masks = np.zeros((self.window_size, 1))
        for index, clipName in enumerate(windowClipNames):
            if clipName != "":
                obj_feature = np.load(self.clips[clipName]["path"] + "_features.npy")
                obj_box = np.load(self.clips[clipName]["path"] + "_boxes.npy")
                # frame_feature = np.load(self.clips[clipName]["path"] + "_img_feature.npy")
                frame_feature = np.load(self.clips[clipName]["path"] + "_i3d.npy")
                # frame_box = np.load(self.clips[clipName]["path"] + "_img_box.npy")
                frame_box = np.zeros(4)
                mask = 1
                anomalies.append(self.clips[clipName]["anomaly"])
                categories.append(self.clips[clipName]["category"])
            else:
                frame_feature = np.zeros((1, 1024, 7, 7))
                frame_box = np.zeros(4)
                # frame_feature = np.zeros((1, self.featureSize))
                obj_feature = np.zeros(self.featureSize) + self.mask_value
                mask = 0
                anomalies.append(-1)
                categories.append(-1)
            obj_features[index, :] = obj_feature
            obj_boxes[index, :] = obj_box
            masks[index, :] = mask
            frame_features[index, :, :] = frame_feature
            frame_boxes[index] = frame_box

        obj_features = torch.from_numpy(obj_features)
        obj_boxes = torch.from_numpy(obj_boxes)
        anomalies = torch.from_numpy(np.array(anomalies))
        frame_features = torch.from_numpy(frame_features)
        frame_boxes = torch.from_numpy(frame_boxes)

        # abnormal video normal clips classified as normal (15)
        # categoriesMerged = np.ones_like(anomalies) * 15
        categories = np.array(categories)
        # categoriesMerged[anomalies == 1] = categories[anomalies == 1]

        categories = torch.from_numpy(np.array(categories))
        masks = torch.from_numpy(np.array(masks))
        return obj_features, obj_boxes, masks, anomalies, categories, windowClipNames, frame_features, frame_boxes

    @staticmethod
    def get_feature(path):
        pass

    @staticmethod
    def collate_fn(data):
        obj_features, obj_boxes, masks, anomalies, categories, windowClipNames, frame_features, frame_boxes = zip(*data)

        obj_features = torch.stack(obj_features, 0)
        obj_boxes = torch.stack(obj_boxes, 0)
        anomalies = torch.stack(anomalies, 0)
        masks = torch.stack(masks, 0)
        categories = torch.stack(categories, 0)
        frame_features = torch.stack(frame_features, 0)
        frame_boxes = torch.stack(frame_boxes, 0)

        return obj_features, obj_boxes, masks, anomalies, categories, windowClipNames, frame_features, frame_boxes


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
