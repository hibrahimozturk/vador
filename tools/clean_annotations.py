import json

new_dict = dict()
with open('data/Annotations/Splits/abnormal/train_val_v1.json') as fp:
    ann_dict = json.load(fp)
    for k, v in ann_dict.items():
        if 'molotof' in k or 'banner' in k:
            continue
        new_dict[k] = v

with open('data/Annotations/Splits/abnormal/train_val_v1_new.json', 'w') as fp:
    json.dump(new_dict, fp)

print('finish')
