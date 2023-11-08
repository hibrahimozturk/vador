from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf
import os
import os.path as osp
import sys
# import numpy as np

train_excludes = ['lr', 'total_loss']
train_freq = 250
val_excludes = ['thr_0.90', 'thr_0.75']
val_freq = 1

current_path = os.getcwd()
si = summary_iterator(osp.realpath(osp.join(current_path, sys.argv[1])))
writer = tf.summary.create_file_writer(osp.join(current_path, sys.argv[2]))

tags = set()
for event in si:
    step = event.step
    for value in event.summary.value:
        tags.add(value.tag)
        if 'val' in value.tag:
            if step % val_freq == 0:
                if value.HasField('simple_value'):
                    include = True
                    for val_exclude in val_excludes:
                        if val_exclude in value.tag:
                            include = False
                    if include:
                        with writer.as_default(step=step):
                            tf.summary.scalar(name=value.tag, data=value.simple_value)
        elif 'train' in value.tag:
            if step % train_freq == 0:
                if value.HasField('simple_value'):
                    include = True
                    for train_exclude in train_excludes:
                        if train_exclude in value.tag:
                            include = False
                    if include:
                        with writer.as_default(step=step):
                            tf.summary.scalar(name=value.tag, data=value.simple_value)
# print(tags)