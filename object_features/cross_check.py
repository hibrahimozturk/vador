import glob
import os
import sys
import json

obj_features = sys.argv[1]
action_features = sys.argv[2]

with open(obj_features) as fp:
    obj_feats = json.load(fp)

names = obj_feats['all_clips']
# action_paths = glob.glob(os.path.join(action_features, '*_i3d.npy'))
# names = dict()
# for action_path in action_paths:
#     name = os.path.basename(action_path).rsplit('_', 1)[0]
#     names[name] = 0

action_paths = glob.glob(os.path.join(action_features, '*_i3d.npy'))
for action_path in action_paths:
    name = os.path.basename(action_path).rsplit('_', 1)[0]
    if name in names:
        del names[name]

print('\n'.join(list(names.keys())))
print('# missed clip features: {}'.format(len(names.keys())))
