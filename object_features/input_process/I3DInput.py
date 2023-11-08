from input_process.InputOutput import InputOutput
import queue
import numpy as np
import torch

import logging
logger = logging.getLogger('extractor')


class I3DInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, inputSize=(224, 224), numThreads=2, max_len=None, flip_frame=False):
        InputOutput.__init__(self, batch, videoPath, annotations, temporalSlide,
                             inputLength, batchSize, 1, inputSize, numThreads, max_len, flip_frame)
        self.required_frames = None
        self.required_index = 0

    def prepare_frame(self, frame):
        return frame

    def prepare_input(self, fps, **kwargs):
        if self.required_frames is None:
            self.required_frames = self.__calculate_required_frames(self.frameCounter-1, self.inputLength, kwargs['total_frames'])
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

    @staticmethod
    def __calculate_required_frames(index, clip_len, end):
        index = max(int(clip_len/2), index)
        index = min(int(end) - int(clip_len/2), index)
        start = index - int(clip_len/2)
        end = index + int(clip_len/2)
        return [start, end]

    def __put_clip(self, video_clip_np, anomaly, frame_index):

        self.inputClips.append(video_clip_np)
        # self.clipNames.append(self.videoName + "_" + str(frame_index).zfill(10) + '_i3d')
        self.clipNames.append(self.videoName + "_" + str(frame_index).zfill(10))
        self.videoNames.append(self.videoName)
        self.targets.append(anomaly)

    @staticmethod
    def __preprocess_input(clipFrames):
        video_clip_np = np.array(clipFrames, dtype='float32')
        video_clip_np = np.interp(video_clip_np, (video_clip_np.min(), video_clip_np.max()), (-1, +1))
        video_clip_np = np.transpose(video_clip_np, (3, 0, 1, 2)).astype(np.float32)
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
