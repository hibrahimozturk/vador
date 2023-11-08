from batch_process.BatchProcessor import BatchProcessor
from models.faster_rcnn_vg.feature_extractor import VideoFasterRCNNFeatureExtractor

import queue
import logging
import numpy as np

logger = logging.getLogger('extractor')


class FasterRCNNProcessor(BatchProcessor):
    def __init__(self, model_cfg, batch: queue.Queue, outputs: queue.Queue, dry_run: bool):
        BatchProcessor.__init__(self, batch, outputs, dry_run)
        self.model = VideoFasterRCNNFeatureExtractor(model_cfg)
        logger.info("faster r-cnn model has been created")

    def process_batch(self):
        if not self.dry_run:
            output = self.model.feature_extract(self.local_batch)
        else:
            output = [dict(
                features=np.zeros((36, 512)),
                boxes=np.zeros((36, 4)),
                img_feature=np.zeros((1024, 7, 7)),
                img_box=np.zeros((1, 4)),
                vis_image=np.zeros((100, 100)),
                vis_image_all=np.zeros((100, 100))) for _ in range(len(self.local_batch))]

        return output

    def put_queue(self, outputs):
        assert len(outputs) == len(self.clip_names) == len(self.video_names) == len(self.targets), \
            "[Faster R-CNN Processor] # of elements are not same"
        for i, output in enumerate(outputs):
            self.outputs.put(dict(
                features=output['features'],
                img_feature=output['img_feature'],
                img_box=output['img_box'],
                boxes=output['boxes'],
                clip_name=self.clip_names[i],
                video_name=self.video_names[i],
                video_info=self.video_info,
                anomaly=self.targets[i],
                vis_image=output['vis_image'],
                vis_image_all=output['vis_image_all']
            ))
