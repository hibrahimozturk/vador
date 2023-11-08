from dataset.ucf_crime_bmn import UCFCrimeBoundary
from utils_.config import Config
from torch.utils.data import DataLoader
from utils_.logger import create_logger

import logging

logger = create_logger("violence")
logger.setLevel(logging.DEBUG)

cfg = Config.fromfile('experiments/ADOR_BMN/configs/dataset_cfg_bmn.py')
cfg.dataset.window_size = 128
cfg.dataset.val.num_keep_objects = 8
t = UCFCrimeBoundary(cfg.dataset, 'train')
dl = DataLoader(t, num_workers=0, batch_size=1, collate_fn=t.collate_fn, shuffle=True)

for b in dl:
    print('.')

a = t[0]

print('finish')
