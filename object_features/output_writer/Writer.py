import os
import json
import queue
import time
from addict import Dict
import numpy as np
import threading
import hashlib

import logging

logger = logging.getLogger('extractor')


class Writer(threading.Thread):
    def __init__(self, outputs: queue.Queue, output_path: str,
                 json_path: str, categories: dict,
                 half_precision: bool, dry_run: bool, pre_json=None):
        threading.Thread.__init__(self)
        self.outputs = outputs
        self.output_path = output_path
        self.json_path = json_path
        self.categories = categories
        self.dry_run = dry_run
        self.clip_labels = dict(
            category_info=dict(),
            abnormal_clips=dict(),
            normal_clips=dict(),
            all_clips=dict(),
            video_clips=dict(),
            video_info=dict()
        )
        self.clip_labels = Dict(self.clip_labels)
        self.output_elements = []
        self.max_write = 24
        self.half_precision = half_precision
        if pre_json is not None:
            self.clip_labels = Dict(pre_json)

        logger.info("writer has been created")

    def run(self):
        info_counter = 0
        while True:
            info_counter += 1
            if self.outputs.qsize() > 0:
                outputs, finish = self.__get_outputs()
                self.output_elements += outputs
                if finish:
                    self.__terminate()
                    break

            if len(self.output_elements) != 0:
                self.__write_outputs()
                self.output_elements = []
            else:
                time.sleep(0.5)
            if info_counter % 50 == 0:
                logger.info("# clip features wait to write: {}".format(self.outputs.qsize()))

    def __get_outputs(self):
        local_outputs = []
        finish = False
        for gi in range(min(self.outputs.qsize(), 24)):
            element = self.outputs.get()
            if element is None:
                finish = True
            local_outputs.append(element)
        return local_outputs, finish

    def __terminate(self):
        with open(self.json_path, "w") as fp:
            json.dump(self.clip_labels, fp, sort_keys=True, indent=4)
        logger.info("output writer has been killed")

    def __write_outputs(self):
        data_type = np.half if self.half_precision else np.single
        for output_element in self.output_elements:

            category = [value for label, value in self.categories.items()
                        if label.lower() in output_element["clip_name"].lower()][0]

            # out_tensor = output_element['out_tensor'].astype(data_type)
            # output_element['out_tensor'] = out_tensor
            #
            # hash = hashlib.md5(out_tensor).hexdigest()
            output_element = self.organize_output_element(output_element, data_type)

            item = dict(clip_name=output_element["clip_name"],
                        category_name=category[0],
                        category=category[1],
                        anomaly=output_element["anomaly"],
                        hash=output_element["hash"])
            self.__append_clip_labels(item, output_element)
            if not self.dry_run:
                self.write_output(output_element)

    def write_output(self, output_element):
        out_tensor = output_element['out_tensor']
        np.savez_compressed(os.path.join(self.output_path, output_element["clip_name"]), out_tensor)

    def organize_output_element(self, output_element, data_type):
        out_tensor = output_element['out_tensor'].astype(data_type)
        output_element['out_tensor'] = out_tensor
        hash = hashlib.md5(out_tensor).hexdigest()
        output_element['hash'] = hash
        return output_element

    def __append_clip_labels(self, item, output_element):
        self.clip_labels.all_clips[output_element["clip_name"]] = item
        if output_element["anomaly"]:
            self.clip_labels.abnormal_clips[output_element["clip_name"]] = item
        else:
            self.clip_labels.normal_clips[output_element["clip_name"]] = item

        if not output_element["video_name"] in self.clip_labels.video_clips:
            self.clip_labels.video_clips[output_element["video_name"]] = []
        self.clip_labels.video_clips[output_element["video_name"]].append(output_element["clip_name"])
        self.clip_labels.video_clips[output_element["video_name"]].sort()

        if not output_element['video_name'] in self.clip_labels.video_info:
            self.clip_labels.video_info[output_element['video_name']] = output_element['video_info']
