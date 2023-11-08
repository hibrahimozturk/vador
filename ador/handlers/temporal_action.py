from handlers.handler import ModelHandler
from abc import ABCMeta, abstractmethod
from addict import Dict
import numpy as np
import os
import os.path as osp
from metrics import f_score, calc_f1
from utils_.utils import visualize_temporal_action
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# import json
import pickle


class TemporalActionSegHandler(ModelHandler):
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        ModelHandler.__init__(self, cfg)
        self.mask_value = cfg.dataset.mask_value
        self.num_classes = 1
        self.iou_list = cfg.evaluation.iou_list
        if cfg.mode == 'train':
            self.exp_dir = cfg.train.exp_dir
            self.visualize_output = cfg.train.visualize
        elif cfg.mode == 'test':
            self.exp_dir = cfg.test.exp_dir
            self.visualize_output = cfg.test.visualize

    # @abstractmethod
    # def get_dataloaders(self, data_cfg, mode):
    #     pass

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    @abstractmethod
    def model_forward(self, data, evaluate=False, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def create_losses(loss_cfg):
        pass

    def init_eval_epoch_dict(self):
        epoch_dict = self.init_epoch_dict()
        val_epoch_dict = Dict(targets=dict(),
                              predictions=dict(),
                              level_preds=dict())
        epoch_dict.update(val_epoch_dict)
        return epoch_dict

    def eval_iteration(self, data, epoch_dict, epoch):
        report, epoch_dict, clip_dicts = super().eval_iteration(data, epoch_dict, epoch)
        # with torch.no_grad():
        #     loss, report, clip_dicts, loss_dict = self.model_forward(data, evaluate=True, epoch=epoch)
        #
        # for loss_key in loss_dict:
        #     if loss_key not in epoch_dict.losses:
        #         epoch_dict.losses[loss_key] = []
        #     epoch_dict.losses[loss_key].append(loss_dict[loss_key])
        #
        # epoch_dict.predictions.update(clip_dicts.predictions)
        # epoch_dict.targets.update(clip_dicts.targets)

        for stage_index, s_clip_dicts in enumerate(clip_dicts):
            if stage_index not in epoch_dict.level_preds:
                epoch_dict.level_preds[stage_index] = dict()
            epoch_dict.level_preds[stage_index].update(s_clip_dicts.predictions)

        return report, epoch_dict, clip_dicts

    def calculate_score(self, epoch, epoch_dict, epoch_report, bg_class=0):
        video_clips, _, _, _ = self.organize_video_clip(epoch_dict)
        if self.visualize_output:
            self.visualize_outputs(epoch, video_clips, self.exp_dir, '.png')
        epoch_report['level_scores'] = dict()
        if 'level_preds' in epoch_dict:
            for level, level_pred in epoch_dict.level_preds.items():
                video_clips_, all_preds, all_gt, _ = self.organize_video_clip(epoch_dict, level)
                all_preds = np.array(all_preds)
                all_preds = all_preds.tolist()
                level_auc = roc_auc_score(all_gt, all_preds)
                # level_auc = average_precision_score(all_gt, all_preds)
                if self.visualize_output:
                    sub_dir = osp.join(str(epoch), 'level_outputs')
                    self.visualize_outputs(sub_dir, video_clips_, self.exp_dir, '_{}.png'.format(level))
                epoch_report['level_scores'][level] = self.temporal_score(self.iou_list, video_clips_, bg_class)
                epoch_report['level_scores'][level]['auc'] = level_auc

        temporal_scores = self.temporal_score(iou_list=self.iou_list,
                                              video_clips=video_clips,
                                              bg_class=bg_class)
        epoch_report['scores'] = temporal_scores
        epoch_report['message'] += self.score_message(temporal_scores, epoch_report['level_scores'])

        return epoch_report, video_clips

    def visualize_outputs(self, sub_dir, video_clips, exp_dir, file_ext='.png'):
        cache_images = osp.join(self.cache_dir, 'output_{}'.format(self.mode), 'images')
        cache_pickles = osp.join(self.cache_dir, 'output_{}'.format(self.mode), 'pickles')

        image_dir = osp.join(exp_dir, 'output_{}'.format(self.mode), 'images')
        pickles_dir = osp.join(exp_dir, 'output_{}'.format(self.mode), 'pickles')

        if not osp.exists(osp.join(exp_dir, 'output_{}'.format(self.mode))):
            os.makedirs(osp.join(exp_dir, 'output_{}'.format(self.mode)))

        for src_path, target_path in zip([pickles_dir, image_dir], [cache_pickles, cache_images]):
            if not osp.exists(src_path):
                if not osp.exists(osp.join(exp_dir, target_path)):
                    os.makedirs(osp.join(exp_dir, target_path))

                os.symlink(osp.join('..', target_path), src_path)

        # if not osp.exists(pickles_dir):
        #     os.makedirs(pickles_dir)

        # if not osp.exists(osp.join(output_dir, 'images')):
        #     os.makedirs(osp.join(output_dir, 'images'))

        image_sub_dir = osp.join(exp_dir, cache_images, str(sub_dir))
        pickels_sub_dir = osp.join(exp_dir, cache_pickles, str(sub_dir))

        for path in [image_sub_dir, pickels_sub_dir]:
            if not osp.exists(path):
                os.makedirs(path)

        self.temporal_visualize(video_clips, image_sub_dir, pickels_sub_dir, file_ext)

    @staticmethod
    def temporal_visualize(video_clips, image_dir, pickels_dir, file_ext):
        output_json = dict()
        for video_name, clips in video_clips.items():
            save_path = osp.join(image_dir, video_name.split(".")[0] + file_ext)
            output_json[video_name] = dict(predictions=(np.array(clips.predictions)*500).astype(np.uint8),
                                           targets=np.array(clips.targets, dtype=np.bool))
            visualize_temporal_action(clips.predictions, clips.targets, save_path, video_name)
        with open(osp.join(pickels_dir, 'clips{}.pickle'.format(file_ext.split('.')[0])), 'wb') as fp:
            pickle.dump(output_json, fp)

    @staticmethod
    def score_message(scores, level_scores):
        message = "\n"
        level_aucs = [data['auc'] for i, data in level_scores.items()]
        message += '#' * 10 + ' AUC ' + "#" * 10 + '\n'
        message += "{}\n".format(', '.join(format(x*100, '.2f') for x in level_aucs))
        for thresh, iou_scores in scores.items():
            for score_set in iou_scores:
                if score_set == 'overall':
                    message += "#" * 10 + " " + thresh + " " + "#" * 10 + "\n"
                else:
                    message += "*" * 10 + " " + score_set + " " + "*" * 10 + "\n"
                for iou_thresh, thresh_scores in iou_scores[score_set].items():
                    level_f1s = [x[thresh][score_set][iou_thresh]['f1'] for i, x in level_scores.items()]
                    message += "{} - f1: {:.2f} [{}]\n".format(iou_thresh, thresh_scores['f1'],
                                                               ', '.join(format(x, '.2f') for x in level_f1s))
        return message

    def organize_video_clip(self, epoch_dict, level=None):
        if level is None:
            predictions = epoch_dict.predictions
        else:
            predictions = epoch_dict.level_preds[level]

        targets = epoch_dict.targets

        video_clips = dict()

        entire_predictions = []
        entire_targets = []

        normal_video_preds = []

        dataloader = getattr(self, "{}_loader".format(self.mode))
        video_clip_list = dataloader.dataset.__getVideoClips__()
        for video_name, clip_list in video_clip_list.items():
            clips = Dict(predictions=[], targets=[])
            for clip_name in clip_list:
                # # TODO: for TALNet, it will be deleted !!!!!!!!!!!!
                # if clip_name not in predictions:
                #     predictions[clip_name] = [0]
                #
                # if clip_name not in targets:
                #     targets[clip_name] = [0]

                clip_pred = np.mean(np.array(predictions[clip_name]))
                clip_gt = np.mean(np.array(targets[clip_name]))

                if 'Normal' in video_name:
                    normal_video_preds.append(clip_pred)

                entire_predictions.append(clip_pred)
                entire_targets.append(clip_gt)

                clips.predictions.append(clip_pred)
                clips.targets.append(clip_gt)

            video_clips[video_name] = clips

        return video_clips, entire_predictions, entire_targets, normal_video_preds

    def temporal_score(self, iou_list, video_clips, bg_class=0):
        output_scores = self.__init_out_score_dict(iou_list)

        for iou in iou_list:
            confusion_mat = Dict(fp=0, tp=0, fn=0)
            class_confusion_mat = Dict()
            for c in range(self.num_classes):
                class_confusion_mat[c] = Dict(fp=0, tp=0, fn=0)
            for video_name, clip_list in video_clips.items():
                clips = video_clips[video_name]
                for c in range(self.num_classes):
                    targets = (np.array(clips.targets) == c)
                    predictions = (np.array(clips.predictions) == c)
                    tp1, fp1, fn1 = f_score(predictions, targets, iou, bg_class=0)

                    class_confusion_mat[c].fp += fp1
                    class_confusion_mat[c].tp += tp1
                    class_confusion_mat[c].fn += fn1

                tp1, fp1, fn1 = f_score(clips.predictions, clips.targets, iou, bg_class=bg_class)

                confusion_mat.tp += tp1
                confusion_mat.fp += fp1
                confusion_mat.fn += fn1

            for c in range(self.num_classes):
                output_scores["class_{}".format(c)]["iou_{:.2f}".format(iou)] = calc_f1(class_confusion_mat[c].fn,
                                                                                        class_confusion_mat[c].fp,
                                                                                        class_confusion_mat[c].tp)

            output_scores.overall["iou_{:.2f}".format(iou)] = calc_f1(confusion_mat.fn,
                                                                      confusion_mat.fp,
                                                                      confusion_mat.tp)


        return output_scores

    def __init_out_score_dict(self, iou_list):
        output_scores = Dict(overall=dict())
        for c in range(self.num_classes):
            output_scores["class_{}".format(c)] = dict()
            for iou in iou_list:
                output_scores["class_{}".format(c)]["iou_{:.2f}".format(iou)] = Dict(f1=0, precesion=0, recall=0)
                output_scores.overall["iou_{:.2f}".format(iou)] = Dict(f1=0, precesion=0, recall=0)
        return output_scores
