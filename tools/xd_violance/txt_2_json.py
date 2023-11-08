import json
import os.path as osp
import cv2

ann_path = '../../data/xd_violance/annotations.txt'
videos_path = '../../data/xd_violance/videos'
output_path = '../../data/xd_violance/test.json'

ann_dict = dict()
with open(ann_path) as fp:
    for line in fp:
        video_arr = []
        parts = line.split()
        assert (len(parts) - 1) % 2 == 0
        video_name = parts.pop(0)

        if video_name[-4:] != '.mp4':
            video_name += '.mp4'
        video_path = osp.join(videos_path, video_name)

        if osp.exists(video_path):
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)

            num_actions = int(len(parts) / 2)
            for action_id in range(num_actions):
                start = int(parts[action_id*2]) / fps
                end = int(parts[action_id*2 + 1]) / fps
                video_arr.append(dict(start=int(start), end=int(end)))

            ann_dict[video_name] = video_arr
        else:
            raise Exception

with open(output_path, 'w') as fp:
    json.dump(ann_dict, fp)

print('finish')