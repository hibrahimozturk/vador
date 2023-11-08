import os
import os.path as osp
import random
import sys
import shutil
from datetime import datetime
from datetime import timedelta
import time
import pickle
import glob
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
import wandb
from wandb import AlertLevel

from handlers.handler import HANDLERS
from utils_.utils import ignore_func
from utils_.config import Config
from utils_.logger import create_logger, logging
from utils_.utils import input_with_timeout, AverageMeter
from utils_.ddp import setup_distributed

logger = create_logger("violence")
logger.setLevel(logging.INFO)


class ViolenceDetection:
    def __init__(self, cfg, local_cfg):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        # logger.info(cfg.pretty_text)
        cfg.start_time = datetime.now().strftime("%d%m%Y-%H%M%S")
        cfg.exp_id = None
        self.user_name = local_cfg.user_name

        self.epoch = -1
        self.step_counter = 0
        self.config = Dict(log_step=cfg.log_step,
                           val_bg_class=cfg.evaluation.bg_class)

        self.distributed = cfg.distributed
        if cfg.distributed:
            cfg.dist_dict = dict(world_size=int(os.environ['WORLD_SIZE']),
                                 proc_id=int(os.environ['LOCAL_RANK']))
            self.config.proc_id = int(os.environ['LOCAL_RANK'])
            print('debug - before ddp init proc_id {} world_size {}'.format(cfg.dist_dict.proc_id, cfg.dist_dict.world_size))
            setup_distributed(cfg.dist_dict.proc_id, cfg.dist_dict.world_size)

        else:
            cfg.dist_dict = None
            self.config.proc_id = 0

        if hasattr(cfg, "train"):
            if 'exp_dir' not in cfg.train:
                if 'exp_name' not in cfg.train:
                    cfg.train.exp_dir = osp.join(osp.dirname(cfg.filename), '{}-{}'.format(self.user_name,
                                                                                           cfg.start_time))
                else:
                    cfg.train.exp_dir = osp.join(osp.dirname(cfg.filename), '{}-{}'.format(self.user_name,
                                                                                           cfg.train.exp_name))

            self.config.exp_dir = cfg.train.exp_dir
            self.num_epochs = cfg.train.num_epochs

            cfg.cache_dir = ''
            cfg.exp_id = ''

            if hasattr(cfg.train, 'checkpoint'):
                cache_dir, exp_id = self.get_cache(cfg, 'train')

                cfg.cache_dir = cache_dir
                cfg.exp_id = exp_id

                self.config.chk_dir = os.path.join(cfg.train.exp_dir, "checkpoints")
                self.config.chk_params = cfg.train.checkpoint.params
                self.config.chk.only_last = cfg.train.checkpoint.only_last

                if not os.path.exists(self.config.chk_dir):
                    big_folder = os.path.join(cfg.cache_dir, 'checkpoints')
                    if self.config.proc_id == 0:
                        if not osp.exists(osp.join(self.config.exp_dir, big_folder)):
                            os.makedirs(osp.join(self.config.exp_dir, big_folder))

                    os.symlink(big_folder, self.config.chk_dir)

            if hasattr(cfg.train, 'visualize') and getattr(cfg.train, 'visualize'):

                cache_dir, exp_id = self.get_cache(cfg, 'train')
                cfg.cache_dir = cache_dir
                cfg.exp_id = exp_id

            if hasattr(cfg.train, "log") and getattr(cfg.train, "log"):
                fh_path = os.path.join(self.config.exp_dir, 'output_train.log')
                if self.config.proc_id == 0:
                    if not os.path.exists(os.path.dirname(fh_path)):
                        os.makedirs(os.path.dirname(fh_path))
                fh = logging.FileHandler(fh_path)
                logger.addHandler(fh)

            if hasattr(cfg.train, 'backup') and getattr(cfg.train, 'backup'):
                if self.config.proc_id == 0:
                    dst_folder = os.path.join(self.config.exp_dir, "backup_code")
                    self.backup_src(dst_folder)

                    with open(os.path.join(self.config.exp_dir, 'train_cfg.py'), 'w') as fp:
                        fp.write(cfg.pretty_text)

            if hasattr(cfg.train, "tensorboard") and getattr(cfg.train, "tensorboard") and self.config.proc_id == 0:
                cache_dir, exp_id = self.get_cache(cfg, 'train')

                cfg.cache_dir = cache_dir
                cfg.exp_id = exp_id

                t_path = os.path.join(cfg.train.exp_dir, 'tensorboard')

                if not os.path.exists(t_path):
                    big_folder = os.path.join(cfg.cache_dir, 'tensorboard')

                    if not osp.exists(osp.join(self.config.exp_dir, big_folder)):
                        os.makedirs(osp.join(self.config.exp_dir, big_folder))

                    os.symlink(big_folder, t_path)

                current_time = datetime.now()
                experiment_id = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(current_time.year, current_time.month,
                                                                       current_time.day, current_time.hour,
                                                                       current_time.minute)

                self.writer = SummaryWriter(log_dir=t_path)
                self.wandb = wandb.init(project='vador', entity="hibrahimozturk", name=cfg.train.exp_name,
                                        id=experiment_id, config=vars(cfg))
                # wandb.config = self.config


        elif hasattr(cfg, "test"):
            if 'exp_dir' not in cfg.test:
                if 'exp_name' not in cfg.test:
                    cfg.test.exp_dir = osp.join(osp.dirname(cfg.filename), '{}-{}'.format(self.user_name,
                                                                                          cfg.start_time))
                else:
                    cfg.test.exp_dir = osp.join(osp.dirname(cfg.filename), '{}-{}'.format(self.user_name,
                                                                                          cfg.test.exp_name))

            cache_dir, exp_id = self.get_cache(cfg, 'test')

            cfg.cache_dir = cache_dir
            cfg.exp_id = exp_id

            self.config.exp_dir = cfg.test.exp_dir

            if hasattr(cfg.test, "log") and getattr(cfg.test, "log") and self.config.proc_id == 0:
                # str_time = datetime.now().strftime("%d%m%Y-%H%M%S")
                str_time = cfg.start_time
                fh_path = os.path.join(self.config.exp_dir, 'output_test.log')
                if not os.path.exists(os.path.dirname(fh_path)):
                    os.makedirs(os.path.dirname(fh_path))
                fh = logging.FileHandler(fh_path)
                logger.addHandler(fh)

            if hasattr(cfg.test, 'visualize') and getattr(cfg.test, 'visualize')  and self.config.proc_id == 0:
                cache_dir, exp_id = self.get_cache(cfg, 'test')
                cfg.cache_dir = cache_dir
                cfg.exp_id = exp_id

            # if hasattr(cfg.test, 'visualize') and getattr(cfg.test, 'visualize'):
            #     # cache_dir, exp_id = self.get_cache(cfg, 'test')
            #     # cfg.cache_dir = cache_dir
            #     # cfg.exp_id = exp_id
            #
            #     visualize_dir = osp.join(cfg.test.exp_dir, 'outputs')
            #
            #     if not osp.exists(visualize_dir):
            #         big_folder = osp.join(cfg.cache_dir, 'outputs')
            #         if not osp.exists(osp.join(self.config.exp_dir, big_folder)):
            #             os.makedirs(osp.join(self.config.exp_dir, big_folder))
            #
            #         os.symlink(big_folder, visualize_dir)

        # if hasattr(cfg, "train"):
        #
        #
        #
        # elif hasattr(cfg, 'test'):

        cfg.proc_id = self.config.proc_id
        self.handler = HANDLERS.get(cfg.model_type)(cfg)

        if hasattr(cfg, 'load_checkpoint'):
            chk_path = os.path.join(self.handler.exp_dir, 'checkpoints', cfg.load_checkpoint)
            self.load_checkpoint(chk_path)
        elif hasattr(cfg, "load_from"):
            self.load_checkpoint(cfg.load_from)

        if cfg.distributed:
            self.handler.move_ddp_model()

        logger.info(cfg.pretty_text)

    def train(self):
        torch.set_num_threads(4)
        self.handler.mode = "train"
        for epoch in range(self.num_epochs):
            self.epoch = epoch

            if self.distributed:
                self.handler.train_loader.sampler.set_epoch(epoch)

            epoch_dict = self.handler.init_train_epoch_dict()
            self.handler.model_params(epoch=epoch)
            self.handler.model.train()
            self.handler.mode = "train"

            batch_time = AverageMeter()
            data_time = AverageMeter()

            end = time.time()
            for step, data in enumerate(self.handler.train_loader):
                data_time.update(time.time() - end)
                iter_report, epoch_dict = self.handler.train_iteration(data, epoch_dict, epoch)
                self.step_counter += 1
                batch_time.update(time.time() - end)
                end = time.time()
                if self.distributed and self.config.proc_id != 0:
                    continue
                info = self.iter_info("train", step, len(self.handler.train_loader), iter_report, batch_time, data_time)
                if step % self.config.log_step == 0:
                    logger.info(info)
                iter_report['epoch'] = epoch
                self.report(iter_report, "train")


            epoch_report = self.handler.epoch_report(epoch_dict)
            self.report(epoch_report, "train/epoch")
            if 'chk_dir' in self.config:
                if self.config.proc_id:
                    if self.config.chk.only_last:
                        [os.remove(os.path.join(self.config.chk_dir, cpath)) for cpath in os.listdir(self.config.chk_dir)]
                    self.save_checkpoint(os.path.join(self.config.chk_dir, "epoch-{}.pth".format(self.epoch)))
            self.handler.end_of_train_epoch()
            self.eval_epoch(epoch)

    def test(self):
        self.eval_epoch(epoch=0, split="test")

    def eval_epoch(self, epoch, split="val"):
        self.handler.mode = split
        # self.handler.mode = 'val'
        self.handler.model.eval()
        epoch_dict = self.handler.init_eval_epoch_dict()
        dataloader = getattr(self.handler, "{}_loader".format(split))

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        for step, data in enumerate(dataloader):
            data_time.update(time.time() - end)
            iter_report, epoch_dict, _ = self.handler.eval_iteration(data, epoch_dict, epoch)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.distributed and self.config.proc_id != 0:
                continue
            info = self.iter_info(split, step, len(dataloader), iter_report, batch_time, data_time)

            if step % self.config.log_step == 0:
                logger.info(info)

        if self.distributed:
            if self.config.proc_id != -1:
                with open('epoch_dict_{}.pickle'.format(self.config.proc_id), 'wb') as fp:
                    pickle.dump(epoch_dict.to_dict(), fp)
            dist.barrier()

            if self.config.proc_id == 0:
                epoch_dict = epoch_dict.to_dict()
                filenames = glob.glob('epoch_dict_*.pickle')
                for filename in filenames:
                    with open(filename, 'rb') as fp:
                        epoch_dict_ = pickle.load(fp)
                        for key, value in epoch_dict_.items():
                            if key == 'losses':
                                for loss_key, loss_values in value.items():
                                    epoch_dict['losses'][loss_key] += loss_values
                            else:
                                epoch_dict[key].update(value)

        epoch_dict = Dict(epoch_dict)
        if self.distributed and self.config.proc_id != 0:
            return
        epoch_report = self.handler.epoch_report(epoch_dict)
        epoch_report, output_data = self.handler.calculate_score(self.epoch, epoch_dict, epoch_report,
                                                                 bg_class=self.config.val_bg_class)

        if hasattr(self, 'wandb'):
            if epoch_report['level_scores'][0]['auc'] > 0.83:
                self.wandb.alert(
                    title="High AUC",
                    text="Accuracy {} is higher than the acceptable threshold".format(epoch_report['level_scores'][0]['auc']*100),
                    level=AlertLevel.WARN,
                    wait_duration=300
                )
        epoch_report['epoch'] = epoch
        self.report(epoch_report, split)

    def report(self, report_data, phase):
        if 'message' in report_data:
            logger.info('-' * 50)
            logger.info("{:5} [{:3}] : {}".format(phase, self.epoch, report_data['message']))
            logger.info('-' * 50)
            del report_data['message']
        if hasattr(self, "writer"):
            for label, data in report_data.items():
                absolute_label = phase + "/" + str(label)
                if type(data) == dict or type(data) == Dict:
                    self.report(data, absolute_label)
                else:
                    self.writer.add_scalar(absolute_label, data, self.step_counter)
                    self.wandb.log({absolute_label: data}, step=self.step_counter)
        # else:
        #     logger.debug("report has not written anywhere because tensorboard was not defined")

    def iter_info(self, phase, step, total_step, report_data, batch_time, data_time):
        eta = str(timedelta(seconds=int((total_step - step) * batch_time.avg)))
        pre_info = "{:5} [{:3}] : [{:3} / {}]  ETA:{} global_step:{:5} batch_time:{:.2f}({:.2f})  data_time:{:.2f}({:.2f}) ".\
            format(phase, self.epoch, step,  total_step,  eta, self.step_counter, batch_time.val, batch_time.avg, data_time.val, data_time.avg)
        info = self.handler.iter_info(report_data)
        return pre_info + info

    def save_checkpoint(self, path):
        save_dict = self.handler.save_variables()
        save_dict.step_counter = self.step_counter
        filtered_dict = dict()
        for param_name in self.config.chk_params:
            filtered_dict[param_name] = save_dict[param_name]
        torch.save(filtered_dict, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if hasattr(checkpoint, 'step_counter'):
            self.step_counter = checkpoint["step_counter"]
        self.handler.load_variables(checkpoint)

    def backup_src(self, dst_folder):
        src_folder = os.path.abspath(__file__).rsplit("/", 1)[0]
        # dst_folder = os.path.join(self.config.exp_dir, "backup_code")

        if os.path.exists(dst_folder):
            answer = input_with_timeout("Do you want to delete already exist backup code / keep old backup?(y/n): ",
                                        0, default_answer='y')
            if answer == "y":
                shutil.rmtree(dst_folder)
            else:
                return

        shutil.copytree(src_folder, dst_folder, ignore=ignore_func)

    def get_cache(self, cfg, mode='train'):

        if not osp.exists(getattr(cfg, mode).exp_dir):
            os.makedirs(getattr(cfg, mode).exp_dir)

        if not osp.exists(osp.join(getattr(cfg, mode).exp_dir, 'exp_id')):
            with open(osp.join(getattr(cfg, mode).exp_dir, 'exp_id'), 'w') as fp:
                fp.write(cfg.start_time)
            exp_id = cfg.start_time
        else:
            with open(osp.join(getattr(cfg, mode).exp_dir, 'exp_id'), 'r') as fp:
                line = fp.read()
                exp_id = line.strip()

        cache_dir = osp.join(*['..'] * (len(getattr(cfg, mode).exp_dir.split('/')) - 1),
                             '.cache/{}'.format(self.user_name), exp_id)
        return cache_dir, exp_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('--cfg', type=str)  # positional argument
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg.mode = "train"
    v = ViolenceDetection(cfg)
    v.train()
