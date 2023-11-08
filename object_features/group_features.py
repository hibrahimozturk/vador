import os
import os.path as osp
import json
import glob
import time

import numpy as np
from tqdm.autonotebook import tqdm
import math
import threading


data_path = '../data/object_features/xd_violance_test/all'
compressed_path = '../data/object_features/xd_violance_test/all_compressed'
json_path = '../data/object_features/xd_violance_test/labels.json'
group_size = 10
temporal_stride = 5

def compress_video_feats(video_name):
    video_outpath = osp.join(compressed_path, video_name)
    if not osp.exists(video_outpath):
        os.makedirs(video_outpath)

    video_clips = annotations['video_clips'][video_name]
    num_groups = math.ceil(len(video_clips) / group_size)
    for group_id in range(num_groups):
        group_clips = video_clips[group_id * group_size : (group_id + 1) * group_size]

        obj_features = []
        boxes = []
        frame_features = []

        for clip in group_clips:
            clip_order = clip.rsplit('_', 1)[-1]

            obj_feature = np.load(osp.join(data_path, video_name + '_features', clip_order + '.npz'))['arr_0']
            box = np.load(osp.join(data_path, video_name + '_box', clip_order + '.npz'))['arr_0']
            frame_feature = np.load(osp.join(data_path, video_name + '_i3d', clip_order + '.npz'))['arr_0']

            obj_features.append(obj_feature)
            boxes.append(box)
            frame_features.append(frame_feature)

        obj_features = np.stack(obj_features)
        boxes = np.stack(boxes)
        frame_features = np.stack(frame_features)

        np.savez_compressed(osp.join(video_outpath, '{:05}-{:05}'.format(group_id * group_size * temporal_stride,
                                                                         (group_id + 1) * group_size * temporal_stride)),
                            boxes=boxes, obj_features=obj_features, frame_features=frame_features)
    return 0

with open(json_path) as fp:
    annotations = json.load(fp)

if not osp.exists(compressed_path):
    os.makedirs(compressed_path)

video_names = list(annotations["video_clips"].keys())
all_threads = []
for video_name in tqdm(video_names):
    threads = threading.enumerate()
    while len(threads) >= 12:
        time.sleep(1)
        threads = threading.enumerate()

    thread = threading.Thread(target = compress_video_feats, args=[video_name])
    all_threads.append(thread)
    thread.start()

for thread in all_threads:
    print('Num of alive threads: {:02}'.format(len(threading.enumerate())), end="\r")
    thread.join()

    # video_outpath = osp.join(compressed_path, video_name)
    # if not osp.exists(video_outpath):
    #     os.makedirs(video_outpath)
    #
    # video_clips = annotations['video_clips'][video_name]
    # num_groups = math.ceil(len(video_clips) / group_size)
    # for group_id in range(num_groups):
    #     group_clips = video_clips[group_id * group_size : (group_id + 1) * group_size]
    #
    #     obj_features = []
    #     boxes = []
    #     frame_features = []
    #
    #     for clip in group_clips:
    #         clip_order = clip.rsplit('_', 1)[-1]
    #
    #         obj_feature = np.load(osp.join(data_path, video_name + '_features', clip_order + '.npz'))['arr_0']
    #         box = np.load(osp.join(data_path, video_name + '_box', clip_order + '.npz'))['arr_0']
    #         frame_feature = np.load(osp.join(data_path, video_name + '_i3d', clip_order + '.npz'))['arr_0']
    #
    #         obj_features.append(obj_feature)
    #         boxes.append(box)
    #         frame_features.append(frame_feature)
    #
    #     obj_features = np.stack(obj_features)
    #     boxes = np.stack(boxes)
    #     frame_features = np.stack(frame_features)
    #
    #     np.savez_compressed(osp.join(video_outpath, '{:05}-{:05}'.format(group_id * group_size * temporal_stride,
    #                                                                      (group_id + 1) * group_size * temporal_stride)),
    #                         boxes=boxes, obj_features=obj_features, frame_features=frame_features)

print('finish')
