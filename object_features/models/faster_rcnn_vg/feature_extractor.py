from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import csv
import torch
import json
from models.faster_rcnn_vg.lib.utils.timer import Timer
from torch.autograd import Variable
from models.faster_rcnn_vg.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from models.faster_rcnn_vg.lib.model.rpn.bbox_transform import clip_boxes
from models.faster_rcnn_vg.lib.model.roi_layers import nms
from models.faster_rcnn_vg.lib.model.rpn.bbox_transform import bbox_transform_inv
from models.faster_rcnn_vg.lib.model.utils.net_utils import save_net, load_net, vis_detections
from models.faster_rcnn_vg.lib.model.utils.blob import im_list_to_blob
from models.faster_rcnn_vg.lib.model.faster_rcnn.vgg16 import vgg16
from models.faster_rcnn_vg.lib.model.faster_rcnn.resnet import resnet
import pdb

from utils_.config import Config

import logging
logger = logging.getLogger('extractor')

from extractors.Extractor import EXTRACTORS

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 18
MAX_BOXES = 18


class VideoFasterRCNNFeatureExtractor:
    def __init__(self, model_cfg):
        classes, fasterRCNN = self.load_model(model_cfg)
        self.model = fasterRCNN
        self.model = self.model.cuda()

        self.model.eval()
        self.classes = classes
        self.model_cfg = model_cfg
        self.counter = 0

    def feature_extract(self, input_batch):
        outputs = []
        with torch.no_grad():
            # input_batch = np.stack(input_batch)
            input_batch = torch.stack(input_batch)
            # for img in input_batch:
            output = self.get_detections_from_im(self.model, self.classes, input_batch, self.model_cfg, conf_thresh=0.3)
            output['boxes'] = output['boxes'] / np.array([output['image_w'], output['image_h'],
                                                          output['image_w'], output['image_h']])
            output['boxes'] = np.clip(output['boxes'], 0.0, 1.0)
            for output_index in range(len(output['boxes'])):
                outputs.append(dict(
                    features=output['features'][output_index],
                    boxes=output['boxes'][output_index],
                    img_feature=output['base_feature'][output_index],
                    img_box=np.array([0, 0, output['image_w'], output['image_h']]),
                    vis_image=output['vis_image'][output_index],
                    vis_image_all=output['vis_image_all'][output_index]),
                )

        return outputs

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
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS[None, :]

        im_shape = im_orig[0].shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

            for im_ori in im_orig:
                im = cv2.resize(im_ori, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
                im_scale_factors.append(im_scale)
                processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_detections_from_im(self, fasterRCNN, classes, im_bgr, args, conf_thresh=0.2):
        """obtain the image_info for each image,
        im_file: the path of the image
        im_bgr: frames of a video

        return: dict of {'image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features'}
        boxes: the coordinate of each box
        """
        # im_bgr = im_bgr[None, :].repeat(10, 0)
        # im_bgr = im_bgr[None, :].repeat(1, 0)
        # initilize the tensor holder here.
        det_tic = time.time()
        all_tic = time.time()

        self.counter += 1
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        if args.cuda > 0:
            cfg.CUDA = True

        if len(im_bgr.shape) == 3:
            # im_bgr = im_bgr[:, :, :, np.newaxis]
            im_bgr = im_bgr[:, :, :, None]
            # im_bgr = np.concatenate((im_bgr, im_bgr, im_bgr), axis=3)
            im_bgr = torch.cat((im_bgr, im_bgr, im_bgr), axis=3)
        im = im_bgr

        vis = True

        # blobs, im_scales = self._get_image_blob(im)
        # blobs = im_list_to_blob(im)
        blobs = im
        """Since the images taken from same video batch implementation is possible """
        # assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs
        # im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], 1]], dtype=np.float32)

        im_info_np = im_info_np.repeat(len(im_bgr), 0)

        # im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_blob
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)

        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
        # pdb.set_trace()

        # the region features[box_num * 2048] are required.
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, pooled_feat, \
        base_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, pool_feat=True, resnet_feat=True)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(*bbox_pred.shape[:2], 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(*bbox_pred.shape[:2], 4 * len(classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # pred_boxes /= im_scales[0]

        # scores = scores.squeeze()
        # pred_boxes = pred_boxes.squeeze()

        det_toc = time.time()


        # detect_time = det_toc - det_tic
        # misc_tic = time.time()

        output_list = dict(features=[], boxes=[], vis_images_all=[],
                           vis_images=[], base_features=[], num_boxes=[])
        for im_index in range(len(im_bgr)):
            max_conf = torch.zeros((pred_boxes.shape[1]))
            if args.cuda > 0:
                max_conf = max_conf.cuda()

            # if vis:
            #     im2show = np.copy(im[im_index])
            selected_classes = torch.nonzero((scores[im_index, :, 1:] > conf_thresh).any(dim=0)).view(-1)
            selected_classes += 1
            for j in selected_classes.tolist():
            # for j in range(1, len(classes)):
                inds = torch.nonzero(scores[im_index, :, j] > conf_thresh).reshape(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[im_index, :, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[im_index, inds, :]
                    else:
                        cls_boxes = pred_boxes[im_index, inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    index = inds[order[keep]]
                    max_conf[index] = torch.where(scores[im_index, index, j] > max_conf[index],
                                                  scores[im_index, index, j], max_conf[index])
                    # if vis:
                    #     im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), conf_thresh)

            if args.cuda > 0:
                keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0).cuda())
            else:
                keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0))
            keep_boxes = torch.squeeze(torch.nonzero(keep_boxes), dim=1)
            if len(keep_boxes) < MIN_BOXES:
                keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
            elif len(keep_boxes) > MAX_BOXES:
                keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

            objects = torch.argmax(scores[im_index, keep_boxes][:, 1:], dim=1)
            box_dets = np.zeros((len(keep_boxes), 4))
            boxes = pred_boxes[im_index, keep_boxes]
            for i in range(len(keep_boxes)):
                kind = objects[i] + 1
                bbox = boxes[i, kind * 4: (kind + 1) * 4]
                box_dets[i] = np.array(bbox.cpu())

            # im2show_1 = np.copy(im[im_index])
            # im2show_1 = vis_detections(im2show_1, 'obj', np.stack(box_dets), conf_thresh)

            output_list['features'].append((pooled_feat[im_index, keep_boxes].cpu()).detach().numpy())
            # output_list['confidences'].append(scores[im_index, keep_boxes].cpu().detach().numpy())
            # output_list['features'].append((pooled_feat[keep_boxes].cpu()).detach().numpy())
            output_list['boxes'].append(box_dets)
            output_list['base_features'].append((base_feat[im_index].cpu()).detach().numpy())
            # output_list['base_features'].append((base_feat.cpu()).detach().numpy())
            output_list['num_boxes'].append(len(keep_boxes))
            # output_list['vis_images'].append(im2show)
            output_list['vis_images'].append(0)
            # output_list['vis_images_all'].append(im2show_1)
            output_list['vis_images_all'].append(0)

        all_toc = time.time()
        logger.debug('Detection time: {:.2f} Resizing time: {:.2f} Ratio: {:.2f}'.format(det_toc-det_tic,
                                                                                         all_toc-all_tic,
                                                                             (det_toc-det_tic)/(all_toc-all_tic)))

        return {
            'image_h': np.size(im, 1),
            'image_w': np.size(im, 2),
            'num_boxes': output_list['num_boxes'],
            'boxes': np.stack(output_list['boxes']),
            'features': output_list['features'],
            'base_feature': output_list['base_features'],
            'vis_image': output_list['vis_images'],
            'vis_image_all': output_list['vis_images_all']
        }

    @staticmethod
    def load_model(args):
        # set cfg according to the dataset used to train the pre-trained model
        if args.dataset == "pascal_voc":
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "pascal_voc_0712":
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "coco":
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "imagenet":
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "vg":
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)

        cfg.USE_GPU_NMS = args.cuda

        print('Using config:')
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)

        # Load classes
        classes = ['__background__']
        with open(os.path.join(args.classes_dir, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                classes.append(object.split(',')[0].lower().strip())

        if not os.path.exists(args.load_dir):
            raise Exception('There is no input directory for loading network from ' + args.load_dir)
        load_name = os.path.join(args.load_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.dataset))

        # initilize the network here. the network used to train the pre-trained model
        if args.net == 'vgg16':
            fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN.create_architecture()

        print("load checkpoint %s" % (load_name))
        if args.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')

        print("load model %s" % (load_name))

        return classes, fasterRCNN


if __name__ == '__main__':

    f = VideoFasterRCNNFeatureExtractor(Config.fromfile('res101_vg_cfg.py'))

    capture = cv2.VideoCapture("/home/halil/Workspace/relation_anomaly/data/Videos/Abuse/Abuse001_x264.mp4")
    batch = []
    batch_size = 4
    i = 0

    while True:
        ret, img = capture.read()
        if not ret:
            break
        batch.append(img)

        i += 1
        if len(batch) == batch_size:
            outputs = f.feature_extract(batch)
            batch = []
