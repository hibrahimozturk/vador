import os
import os.path as osp
import glob
import numpy as np

video_name = 'RoadAccidents044_x264_features'

path1 = '../data/object_features/train_exp/abnormal'
path2 = '../data/object_features/train_5fps/abnormal'

for feat_name in sorted(os.listdir(osp.join(path1, video_name))):
    feat1 = np.load(osp.join(path1, video_name, feat_name))
    feat2 = np.load(osp.join(path2, video_name, feat_name))
    diff = feat1['arr_0'] - feat2['arr_0']
    if np.absolute(diff).mean() > 1e-5 or np.absolute(diff).var() > 1e-5:
        print('.')
