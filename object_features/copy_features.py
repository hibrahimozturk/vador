import os
import glob
import sys
import shutil
import tqdm

classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Banner', 'Burglary', 'Explosion', 'Fighting', 'MolotovBomb',
           'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

src_path = sys.argv[1]
tgt_path = sys.argv[2]

for class_name in classes:
    print(class_name)
    feature_path = os.path.join(src_path, class_name, 'abnormal', '*.npy')
    for path in tqdm.tqdm(glob.glob(feature_path)):
        shutil.copy(path, tgt_path)
