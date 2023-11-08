from utils_.logger import create_logger
from utils_.config import Config
import logging
import sys
import argparse

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')
parser.add_argument('--cfg', type=str)  # positional argument
parser.add_argument('--local_rank', type=int, default=0)  # positional argument
parser.add_argument('--distributed', action='store_true')  # positional argument
args = parser.parse_args()

cfg = Config.fromfile(args.cfg)
cfg.distributed = args.distributed
if hasattr(cfg, 'python_path'):
    cfg.train.backup = False
    filtered_paths = [path for path in sys.path if path.rsplit('/', 1)[-1] != 'vador']
    sys.path = filtered_paths
    sys.path.insert(0, cfg.python_path)

from violance_v2 import ViolenceDetection

logger = create_logger("violence")
logger.setLevel(logging.DEBUG)

local_cfg = Config.fromfile('config.py')
vlcdtr = ViolenceDetection(cfg, local_cfg)
# vlcdtr = ViolenceDetection(cfg)
vlcdtr.train()
