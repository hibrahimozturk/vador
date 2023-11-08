import torch.nn as nn
import torch
import numpy as np
import itertools


class Anchors1D(nn.Module):
    def __init__(self, anchor_scale=8., pyramid_levels=None, scale_ranges=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [0, 1, 2, 3, 4, 5]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0)])
        self.scale_ranges = scale_ranges

        self.last_anchors = {}
        self.last_scales = {}
        self.last_shape = None

    def forward(self, window, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        window_length = window.shape[1:]

        if window_length == self.last_shape and window.device in self.last_anchors:
            return self.last_anchors[window.device], self.last_scales[window.device]

        if self.last_shape is None or self.last_shape != window_length:
            self.last_shape = window_length

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        ranges_all = []
        for i, stride in enumerate(self.strides):
            boxes_level = []
            ranges_level = []
            for _, (scale, ratio) in enumerate(itertools.product(self.scales, self.ratios)):
                if window_length[0] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                # anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, window_length[0], stride)
                # y = np.arange(stride / 2, image_shape[0], stride)
                # xv, yv = np.meshgrid(x, y)
                # xv = xv.reshape(-1)
                # yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.array([x - anchor_size_x_2, x + anchor_size_x_2])
                # boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                #                    yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
                ranges = np.array(self.scale_ranges[i])[None, :].repeat(len(boxes), axis=0)
                ranges_level.append(ranges)

            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 2]))
            ranges_all.append(np.concatenate(ranges_level, axis=0))

        anchor_boxes = np.vstack(boxes_all)
        scale_ranges = np.vstack(ranges_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(window.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        scale_ranges = torch.from_numpy(scale_ranges.astype(dtype)).to(window.device)
        scale_ranges = scale_ranges.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[window.device] = anchor_boxes
        self.last_scales[window.device] = scale_ranges
        return anchor_boxes, scale_ranges
