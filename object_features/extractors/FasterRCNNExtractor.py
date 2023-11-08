from extractors.Extractor import Extractor
from input_process.FasterRCNNInput import FasterRCNNInput
from batch_process.FasterRCNNProcessor import FasterRCNNProcessor
from output_writer.FasterRCNNWriter import FasterRCNNWriter
from extractors.Extractor import EXTRACTORS

import os.path as osp
import logging

logger = logging.getLogger('extractor')


@EXTRACTORS.register_module(name="faster_rcnn")
class FasterRCNNExtractor(Extractor):
    def __init__(self, cfg):
        Extractor.__init__(self, cfg)
        logger.info('faster_rcnn extractor has been created')

        if self.pre_json is not None:
            for clip_name in self.pre_json['all_clips']:
                video_name, clip_oder = clip_name.rsplit('_', 1)
                feat_path = osp.join(cfg.output_writer.clip_folder, video_name + '_features')
                box_path = osp.join(cfg.output_writer.clip_folder, video_name + '_box')
                if not osp.exists(osp.join(feat_path, clip_oder + '.npz')) or \
                        not osp.exists(osp.join(box_path, clip_oder + '.npz')):
                    raise Exception('{} clip does not exists in pre json'.format(clip_name))

    def get_consumer(self, **kwargs):
        consumer = FasterRCNNProcessor(kwargs["cfg"], self.batch, self.outputs, self.dry_run)
        return consumer

    def get_producer(self, video_path, annotations, **kwargs):
        if type(kwargs["cfg"].input_size) == list:
            kwargs["cfg"].input_size = tuple(kwargs["cfg"].input_size)

        producer = FasterRCNNInput(self.batch, video_path, annotations,
                                   kwargs["cfg"].batch_size, kwargs["cfg"].frame_sample_rate, kwargs["cfg"].input_size,
                                   kwargs["cfg"].num_threads, kwargs["cfg"].max_len, kwargs["cfg"].flip_frame)
        return producer

    def get_writer(self, cfg):
        writer = FasterRCNNWriter(self.outputs, cfg.output_writer.clip_folder,
                                  cfg.output_writer.json_path, cfg.extractor.categories,
                                  cfg.output_writer.half_precision, self.dry_run, self.pre_json)
        return writer
