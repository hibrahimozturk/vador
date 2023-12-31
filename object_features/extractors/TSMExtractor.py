from extractors.Extractor import Extractor
from input_process.TSMInput import TSMInput
from batch_process.TSMProcessor import TSMProcessor
from extractors.Extractor import EXTRACTORS

import logging
logger = logging.getLogger('extractor')


@EXTRACTORS.register_module(name="tsm")
class TSMExtractor(Extractor):
    def __init__(self, cfg):
        Extractor.__init__(self, cfg)
        logger.info('tsm extractor has been created')

    def get_consumer(self, **kwargs):
        temporal_stride = 64 // kwargs["cfg"].input_length
        consumer = TSMProcessor(kwargs["cfg"].path, temporal_stride, self.batch, self.outputs, self.dry_run)
        return consumer

    def get_producer(self, video_path, annotations, **kwargs):
        if type(kwargs["cfg"].input_size) == list:
            kwargs["cfg"].input_size = tuple(kwargs["cfg"].input_size)

        producer = TSMInput(self.batch, video_path, annotations, kwargs["cfg"].input_length,
                            kwargs["cfg"].batch_size, kwargs["cfg"].input_size, kwargs["cfg"].input_size,
                            kwargs["cfg"].num_threads, kwargs["cfg"].max_len, kwargs["cfg"].flip_frame)
        return producer
