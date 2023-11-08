from extractors.Extractor import Extractor, EXTRACTORS
from input_process.ResNetI3DInput import ResNetI3DInput
from batch_process.ResNetI3DProcessor import ResNetI3DProcessor
from output_writer.I3DWriter import I3DWriter

import os.path as osp
import logging

logger = logging.getLogger('extractor')


@EXTRACTORS.register_module(name="resnet_i3d")
class ResNetI3DExtractor(Extractor):
    def __init__(self, cfg):
        Extractor.__init__(self, cfg)
        logger.info('resnet i3d extractor has been created')

        if self.pre_json is not None:
            for clip_name in self.pre_json['all_clips']:
                video_name, clip_oder = clip_name.rsplit('_', 1)
                clip_path = osp.join(cfg.output_writer.clip_folder, video_name + '_i3d')
                if not osp.exists(osp.join(clip_path, clip_oder + '.npz')):
                    raise Exception('{} clip does not exists in pre json'.format(clip_name))

    def get_producer(self, video_path, annotations, **kwargs):
        producer = ResNetI3DInput(self.batch, video_path, annotations, kwargs["cfg"].temporal_stride,
                            kwargs["cfg"].input_length,
                            kwargs["cfg"].batch_size, kwargs["cfg"].input_size, kwargs["cfg"].num_threads,
                            kwargs["cfg"].max_len, kwargs["cfg"].flip_frame)
        return producer

    def get_consumer(self, **kwargs):
        consumer = ResNetI3DProcessor(kwargs["cfg"].path,
                                      self.batch, self.outputs, self.dry_run)
        return consumer

    def get_writer(self, cfg):
        writer = I3DWriter(self.outputs, cfg.output_writer.clip_folder,
                           cfg.output_writer.json_path, cfg.extractor.categories,
                           cfg.output_writer.half_precision, self.dry_run, self.pre_json)
        return writer
