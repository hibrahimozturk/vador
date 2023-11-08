from dataset.ucf_crime_objects import UCFCrimeObjects
from utils_.config import Config

cfg = Config.fromfile('experiments/ADOR_MF/configs/dataset_cfg.py')
t = UCFCrimeObjects(cfg.dataset, 'train')
a = t[0]

print('finish')
