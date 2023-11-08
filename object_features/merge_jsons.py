import json

train_json = '../data/object_features/train_5fps/normal_labels.json'
val_json = '../data/object_features/val_5fps/normal_labels.json'
out_json = '../data/object_features/train_5fps/normal_labels_trainval.json'

with open(train_json) as fp:
    train_anns = json.load(fp)

with open(val_json) as fp:
    val_anns = json.load(fp)

for key in train_anns:
    train_anns[key].update(val_anns[key])
# train_anns.update(val_anns)
with open(out_json, 'w') as fp:
    json.dump(train_anns, fp)