import os
import time

import queue
import re
import json
from output_writer.Writer import Writer
from abc import ABCMeta, abstractmethod
from utils_.registry import Registry

import logging

logger = logging.getLogger('extractor')
EXTRACTORS = Registry("extractors")


class Extractor:
    def __init__(self, cfg):
        __metaclass__ = ABCMeta

        logger.info(cfg.pretty_text)

        self.num_producers = cfg.extractor.num_producers
        self.dry_run = cfg.extractor.dry_run
        self.top_k = cfg.extractor.top_k if hasattr(cfg.extractor, "top_k") else None

        self.pre_json = None
        if hasattr(cfg.extractor, 'pre_json'):
            with open(cfg.extractor.pre_json, 'r') as fp:
                self.pre_json = json.load(fp)

        self.batch = queue.Queue(10)
        self.outputs = queue.Queue()

        self.producers = []

        self.writer = self.get_writer(cfg)
        self.consumer = self.get_consumer(cfg=cfg.model)
        self.producer_cfg = cfg.input_processor

        self.video_folder = cfg.extractor.video_folder
        with open(cfg.extractor.temporal_annotions) as fp:
            self.temporal_annotations = json.load(fp)

        if not hasattr(cfg.extractor, 'filter_pattern'):
            self.filter_regex = re.compile(r'.*')
        else:
            self.filter_regex = re.compile(cfg.extractor.filter_pattern)

    @abstractmethod
    def get_consumer(self, **kwargs):
        pass

    @abstractmethod
    def get_producer(self, video_path, annotations, **kwargs):
        pass

    def get_writer(self, cfg):
        writer = Writer(self.outputs, cfg.output_writer.clip_folder,
                        cfg.output_writer.json_path, cfg.extractor.categories,
                        cfg.output_writer.half_precision, self.dry_run, self.pre_json)
        return writer

    def __call__(self):
        logger.debug("extraction process starts")
        num_videos = len(list(filter(self.filter_regex.match, list(self.temporal_annotations.keys()))))
        num_videos = max(num_videos, self.top_k if self.top_k else 0)
        self.consumer.start()
        self.writer.start()
        counter = 0
        for indx, (video_name, annotations) in enumerate(self.temporal_annotations.items()):
            if self.pre_json is not None:
                if os.path.basename(video_name).rsplit('.')[0] in self.pre_json['video_info']:
                    logger.info("{} exists".format(video_name))
                    counter += 1
                    continue
            video_path = os.path.join(self.video_folder, video_name)
            if self.filter_regex.match(video_path):
                if counter == self.top_k:
                    logger.info("top {} videos has been processed".format(self.top_k))
                    break
                logger.info("{:.2f}% percent video has been completed".format((counter / num_videos)*100))
                producer = self.get_producer(video_path, annotations, cfg=self.producer_cfg)
                counter += 1
                producer.start()
                time.sleep(1)
                self.producers.append(producer)
                self.__wait_producers()
        self.__finalize()
        logger.debug("extraction process has been finished")

    def __wait_producers(self):
        while len(self.producers) == self.num_producers:
            tempList = []
            for producer in self.producers:
                if producer.is_alive() is True:
                    tempList.append(producer)
            self.producers = tempList
            logger.debug("# producer threads {}".format(len(self.producers)))
            time.sleep(1)

    def __finalize(self):
        for producer in self.producers:
            producer.join()

        logger.debug("finalize signal to batch queue")
        time.sleep(5)
        self.batch.put(None)
        self.consumer.join()
        self.writer.join()
