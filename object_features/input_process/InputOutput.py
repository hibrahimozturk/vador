import queue
import cv2
import torch
import time
import threading
import numpy as np
from abc import ABCMeta, abstractmethod
import os.path as osp

import logging
logger = logging.getLogger('extractor')


class InputOutput(threading.Thread):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, frameSampleRate=1, inputSize=(224, 224), numThreads=2,
                 max_len=None, flip_frame=False):
        __metaclass__ = ABCMeta
        threading.Thread.__init__(self)

        self.batch = batch
        self.videoPath = videoPath
        # self.videoName = self.videoPath.split("/")[-1].rsplit(".", 1)[0]
        self.videoName = osp.basename(self.videoPath).rsplit(".", 1)[0]
        self.annotations = annotations
        self.temporalSlide = temporalSlide
        self.frameSampleRate = frameSampleRate
        self.inputLength = inputLength
        self.batchSize = batchSize
        self.inputSize = inputSize
        self.numThreads = numThreads
        self.max_len = max_len
        self.flip_frame = flip_frame

        self.frames = []
        self.inputClips = []
        self.clipNames = []
        self.targets = []
        self.videoNames = []

        self.frameCounter = 0
        self.clipFrame = 0

        self.num_frames = None
        self.fps = None

        logger.info("clips of {} are extracting".format(videoPath.split("/")[-1]))

    def run(self):
        cv2.setNumThreads(self.numThreads)
        capture = cv2.VideoCapture(self.videoPath)
        start_t = time.time()
        frame_count = 0
        while True:
            ret = capture.grab()
            if not ret:
                break
            frame_count += 1
        logger.info('counting time: {}'.format(time.time() - start_t))
        capture = cv2.VideoCapture(self.videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)
        # total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = frame_count
        if self.max_len is not None:
            total_frames = min(total_frames, self.max_len + 1)
        self.num_frames = frame_count
        self.fps = fps

        # if self.max_len is not None:
        #     if total_frames > self.max_len:
        #         return 0

        while True:
            if self.max_len is not None:
                if self.frameCounter > self.max_len:
                    if len(self.inputClips) != 0:
                        self.put_queue()
                    logger.info("{} has been finished since max length exceeds".format(self.videoPath.split("/")[-1]))
                    return 0

            ret, img = capture.read()
            if not ret:
                if len(self.inputClips) != 0:
                    self.put_queue()
                logger.info("{} has been finished".format(self.videoPath.split("/")[-1]))
                return 0

            self.frameCounter += 1
            if (self.frameCounter - 1) % self.frameSampleRate == 0:

                if self.inputSize is not None:
                    img = cv2.resize(img, self.inputSize)

                if self.flip_frame:
                    img = cv2.flip(img, 1)
                img = self.prepare_frame(img)

                self.frames.append(img)
                self.prepare_input(fps, total_frames=total_frames)

                self.__queue_full()
                if len(self.inputClips) == self.batchSize:
                    logger.debug("targets: {}".format(self.targets))
                    self.put_queue()
                    logger.debug("batch size: {} (new batch)".format(self.batch.qsize()))
                # TODO: last clips are lost, solve

        return 0

    def put_queue(self):
        assert len(self.inputClips) == len(self.clipNames) == len(self.videoNames) == len(self.targets), \
            "# of elements are not same"
        self.inputClips = self.inputClips
        self.batch.put({"inputClip": self.inputClips,
                        "clipName": self.clipNames,
                        "videoName": self.videoNames,
                        "target": self.targets,
                        "batchSize": len(self.inputClips),
                        "videoInfo": dict(num_frames=self.num_frames,
                                          fps=self.fps,
                                          num_seconds=(self.num_frames/self.fps),
                                          annotations=self.annotations)})
        self.inputClips, self.clipNames, self.targets, self.videoNames = [], [], [], []

    @abstractmethod
    def prepare_input(self, fps, **kwargs):
        pass

    @abstractmethod
    def prepare_frame(self, frame):
        pass

    def __queue_full(self):
        while self.batch.full():
            logger.debug("batch size: {} (full)".format(self.batch.qsize()))
            time.sleep(2)
