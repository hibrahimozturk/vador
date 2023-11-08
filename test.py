from violance_v2 import ViolenceDetection
from utils_.config import Config
from utils_.logger import create_logger
import sys
import argparse

import logging
logger = create_logger("violence")
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')
parser.add_argument('--cfg', type=str)  # positional argument
parser.add_argument('--local_rank', type=int, default=0)  # positional argument
parser.add_argument('--distributed', action='store_true')  # positional argument
args = parser.parse_args()

# TODO: set pythonpath to backup code folder
cfg = Config.fromfile(args.cfg)
cfg.distributed = args.distributed
local_cfg = Config.fromfile('config.py')

cfg.mode = "test"
v = ViolenceDetection(cfg, local_cfg)
# v = ViolenceDetection(cfg)
v.test()
