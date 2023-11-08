import numpy as np
import os.path as osp
import os
import json
from tqdm.autonotebook import tqdm

xdv_path = '../../data/object_features/xd_violance_resnet'
# json_paths = [osp.join(xdv_path, 'abnormal_labels_i3d.json'), osp.join(xdv_path, 'normal_labels_i3d.json')]
json_paths = [osp.join(xdv_path, 'labels_i3d.json')]
video_list = []
all_gts = []

for json_path in json_paths:
    with open(json_path, 'r') as fp:
        anns = json.load(fp)

    for video_name, clip_list in tqdm(anns['video_clips'].items()):
        video_list.append(video_name + '_i3d')
        feats = []
        gts = []
        clip_dict = dict()
        for clip_name_crop in clip_list:
            clip_name, crop_order = clip_name_crop.rsplit('_', 1)
            clip_path = osp.join(xdv_path, 'all', video_name + '_i3d', '_'.join(clip_name_crop.split('_')[2:]) + '.npz')
            clip_feat = np.load(clip_path)['arr_0']
            feats.append(clip_feat)
            gts.append(anns['all_clips'][clip_name_crop]['anomaly'])

        video_feats = np.stack(feats)[:, None, :]
        num_feats = len(video_feats)
        video_feats = video_feats.reshape(int(num_feats / 10), 10, -1)
        video_gts = np.array(gts)
        all_gts.append(video_gts)

        feat_path = osp.join(xdv_path, 'v3', video_name + '_i3d')
        np.save(feat_path, video_feats)


with open(osp.join(xdv_path, 'ucf-i3d-train-10crop.list'), 'w') as fp:
    for line in video_list:
        fp.write(line + '\n')

np.save(osp.join(xdv_path, 'gt-ucf'), np.concatenate(all_gts))
print('finish')

