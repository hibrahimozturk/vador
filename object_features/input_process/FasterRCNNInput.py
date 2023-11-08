from input_process.InputOutput import InputOutput
import queue
import copy
import numpy as np
import cv2
import torch

import logging
logger = logging.getLogger('extractor')


class FasterRCNNInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, batchSize: int,
                 frameSampleRate: int, inputSize=(224, 224), numThreads=2, max_len=None, flip_frame=False):
        InputOutput.__init__(self, batch, videoPath, annotations, -1,
                             1, batchSize, frameSampleRate, inputSize, numThreads, max_len, flip_frame)

        self.frameTargets = []
        self.featureNames = []

    def prepare_frame(self, frame):
        im, im_scale = self._get_image_blob(frame)
        return im

    @staticmethod
    def _get_image_blob(im):
        """Converts an image into a network input.
      Arguments:
        im (ndarray): a color image in BGR order
      Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
          in the image pyramid
      """
        # im_orig = im.astype(np.float32, copy=True)
        im_orig = im.astype(np.float32)
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        im_orig -= pixel_means

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        test_scales = [600]
        test_max_size = 1000
        for target_size in test_scales:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > test_max_size:
                im_scale = float(test_max_size) / float(im_size_max)

            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        return im, im_scale_factors[0]

    def prepare_input(self, fps, **kwargs):
        self.__is_abnormal(fps)
        if len(self.frames) == self.inputLength:
            img = copy.deepcopy(self.frames[0])
            img = torch.from_numpy(img).cuda()
            self.inputClips.append(img)
            # self.inputClips.append(copy.deepcopy(self.frames[0]))
            self.clipNames.append(copy.deepcopy(self.featureNames[0]))
            self.videoNames.append(self.videoName)
            self.targets.append(copy.deepcopy(self.frameTargets[0]))
            self.frames, self.featureNames, self.frameTargets = [], [], []
        return

    def __is_abnormal(self, fps):
        anomaly = 0
        for actionSpace in self.annotations:
            if actionSpace["start"] * fps < self.frameCounter - 1 < actionSpace["end"] * fps:
                anomaly = 1
                break
            else:
                anomaly = 0
        self.frameTargets.append(anomaly)
        self.featureNames.append(self.videoName + "_" + str(self.frameCounter - 1).zfill(10))
