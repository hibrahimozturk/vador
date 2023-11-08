import sys
from extractors.Extractor import EXTRACTORS
from utils_.config import Config
from utils_.logger import create_logger
import logging

logger = create_logger("extractor")
logger.setLevel(logging.DEBUG)

cfg = Config.fromfile(sys.argv[1])
ext = EXTRACTORS.get(cfg.extractor_type)(cfg)
ext()
