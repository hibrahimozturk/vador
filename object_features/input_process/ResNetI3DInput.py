from input_process.InputOutput import InputOutput
import queue
import numpy as np
import torch
import torchvision.transforms as transforms
import utils_.gtransforms as gtransforms
from PIL import Image
import cv2

import logging
logger = logging.getLogger('extractor')


class ResNetI3DInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, inputSize=(224, 224), numThreads=2, max_len=None, flip_frame=False):
        InputOutput.__init__(self, batch, videoPath, annotations, temporalSlide,
                             inputLength, batchSize, 1, inputSize, numThreads, max_len, flip_frame)
        self.required_frames = None
        self.required_index = 0
        mean = [114.75, 114.75, 114.75]
        std = [57.375, 57.375, 57.375]
        self.transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        # gtransforms.GroupCenterCrop(224),
                        gtransforms.GroupTenCrop(224),
                        # gtransforms.ToTensor(),
                        # gtransforms.GroupNormalize(mean, std),
                        gtransforms.GroupNormalize_ten(mean, std),
                        # gtransforms.LoopPad(16),
            ])

    def prepare_frame(self, frame):
        return frame

    def prepare_input(self, fps, **kwargs):
        if self.required_frames is None:
            # self.required_frames = self.__calculate_required_frames(self.frameCounter-1, self.inputLength, kwargs['total_frames'])
            self.required_frames = []
            for index in range(0, kwargs['total_frames'], self.temporalSlide):
                self.required_frames.append(self.__calculate_required_frames(index, self.inputLength, kwargs['total_frames']))
            # self.required_index = self.frameCounter - 1

        if self.required_index <  len(self.required_frames):
            while len(self.frames) == self.required_frames[self.required_index][1]:
                clip_start, clip_end = self.required_frames[self.required_index][0:2]
                anomaly = self.__is_abnormal(clip_start, clip_end, fps, 0.7)

                # video_clip_np = self.__preprocess_input(self.frames[0:self.inputLength])
                video_clip_np = self.__preprocess_input(self.frames[clip_start:clip_end])
                self.__put_clip(video_clip_np, anomaly, self.required_index*self.temporalSlide)

                # self.frames = self.frames[self.temporalSlide:self.inputLength]
                # self.required_index += self.temporalSlide
                self.required_index += 1
                # self.required_frames = self.__calculate_required_frames(self.required_index, self.inputLength,
                #                                                         kwargs['total_frames'])

                if self.required_index >= len(self.required_frames):
                    break
                flush_index = max(min(self.required_frames[self.required_index][0]-1, len(self.frames)), 0)
                self.frames[:flush_index] = [None] * flush_index

        assert len(self.frames) == self.frameCounter

    # @staticmethod
    # def __calculate_required_frames(index, clip_len, end):
    #     index = max(int(clip_len/2), index)
    #     index = min(int(end) - int(clip_len/2), index)
    #     start = index - int(clip_len/2)
    #     end = index + int(clip_len/2)
    #     return [start, end]

    @staticmethod
    def __calculate_required_frames(index, clip_len, end):
        index = max(0, index)
        index = min(int(end) - int(clip_len), index)
        start = index
        end = index + int(clip_len)
        return [start, end]

    def __put_clip(self, video_clip_np, anomaly, frame_index):

        for crop_index in range(len(video_clip_np)):
            self.inputClips.append(video_clip_np[crop_index])
            # self.clipNames.append(self.videoName + "_" + str(frame_index).zfill(10) + '_i3d')
            self.clipNames.append(self.videoName + "_" + str(frame_index).zfill(10)  + '_' + str(crop_index).zfill(2))
            self.videoNames.append(self.videoName)
            self.targets.append(anomaly)

    def __preprocess_input(self, clipFrames):
        frames = []
        for frame in clipFrames:
            # cv2.imshow('abc', frame)
            # ch = cv2.waitKey(1)
            img = Image.fromarray(frame[:, :, ::-1])
            frames.append(img)

        video_clip_np = self.transform(frames)
        # video_clip_np = video_clip_np.permute(1, 0, 2, 3).numpy()
        video_clip_np = torch.stack(video_clip_np, 0).permute(1, 2, 0, 3, 4).numpy()
        # video_clip_np = np.array(clipFrames, dtype='float32')
        # video_clip_np = (video_clip_np * 2) / 255 - 1
        # video_clip_np = np.interp(video_clip_np, (video_clip_np.min(), video_clip_np.max()), (-1, +1))
        # video_clip_np = np.transpose(video_clip_np, (3, 0, 1, 2)).astype(np.float32) # (C, T, H, W)
        return video_clip_np

    def __is_abnormal(self, start, end, fps, intersectionThreshold):
        anomaly = 0
        for actionSpace in self.annotations:
            intersectionEnd = min(end, actionSpace["end"] * fps)
            intersectionStart = max(start, actionSpace["start"] * fps)
            if (intersectionEnd - intersectionStart) / self.inputLength > intersectionThreshold:
                anomaly = 1
                break
            else:
                anomaly = 0
        return anomaly
