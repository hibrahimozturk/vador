import numpy as np
from output_writer.Writer import Writer
import queue
import os
import os.path as osp
import cv2
import hashlib


class I3DWriter(Writer):
    def __init__(self, outputs: queue.Queue, output_path: str,
                 json_path: str, categories: dict, half_precision: bool,
                 dry_run: bool, pre_json=None):
        Writer.__init__(self, outputs, output_path, json_path, categories, half_precision, dry_run, pre_json)

    def write_output(self, output_element):
        out_feat_dir = osp.join(self.output_path, output_element['video_name'] + '_i3d')

        if not osp.exists(out_feat_dir):
            os.makedirs(out_feat_dir)

        frame_order = '_'.join(output_element['clip_name'].split('_')[-1:])
        # base_name = osp.join(out_dir, output_element["clip_name"])
        feat_name = osp.join(out_feat_dir, frame_order)

        np.savez_compressed(feat_name, output_element["out_tensor"])
        # np.save(base_name + "_img_feature", output_element["img_feature"])
        # np.save(base_name + "_img_box", output_element["img_box"])
        # cv2.imwrite(base_name + "_img.jpg", output_element['vis_image'])
        # cv2.imwrite(base_name + "_img_all.jpg", output_element['vis_image_all'])
