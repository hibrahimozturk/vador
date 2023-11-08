import os.path as osp
import os
import glob
import numpy as np
import shutil

# for i in range(1000):
for i in [71]:
    clip_paths = list(glob.glob('../data/object_features/train/abnormal/Fighting0{:02d}*i3d.npy'.format(i)))
    clip_paths.sort()

    if len(clip_paths) != 0:
        video_clips = []
        for clip_path in clip_paths:
            video_clip = np.load(clip_path)
            video_clips.append(video_clip)
            shutil.copyfile(clip_path, osp.join('compressed_outputs/clips', osp.basename(clip_path)))
            np.savez_compressed(osp.join('compressed_outputs/clips_compressed', osp.basename(clip_path).split('.')[0]),
                                video_clip)

        video_clips = np.stack(video_clips)
        np.savez_compressed('compressed_outputs/compressed/Fighting0{:02d}'.format(i), video_clips)
        print('finish')

