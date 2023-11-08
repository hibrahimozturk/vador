import copy
from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from utils_.registry import Registry
from addict import Dict
from dataset.ucf_crime import DATASETS
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils_.warmup_scheduler import GradualWarmupScheduler
from dataset.sampler import SamplerFactory
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_.ddp import reduce_tensor
from torch.optim.lr_scheduler import  CosineAnnealingLR


import logging

logger = logging.getLogger("violence")

HANDLERS = Registry("handlers")


class ModelHandler:
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        self.cfg = cfg
        self.mode = cfg.mode
        self.start_time = cfg.start_time
        self.cache_dir = cfg.cache_dir

        logger.info("model handler switched to {} mode".format(self.mode))

        self.model = self.get_model(cfg.model)
        # if cfg.distributed:
        #     device = torch.device("cuda", cfg.dist_dict.proc_id)
        #     self.model = self.model.to(device)
        #     self.model = DDP(self.model, device_ids=[cfg.dist_dict.proc_id],
        #                      output_device=cfg.dist_dict.proc_id, find_unused_parameters=True)

        self.losses = self.create_losses(cfg.model.losses)

        if self.mode == "train":
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay)

            self.train_loader, self.val_loader = self.get_dataloaders(cfg.dataset, splits=['train', 'val'], dist_cfg=cfg.dist_dict)

            if hasattr(cfg.train.optimizer, "scheduler"):
                # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                #                                                  step_size=cfg.train.optimizer.scheduler.step_size,
                #                                                  gamma=cfg.train.optimizer.scheduler.gamma)
                warmup_epochs = 5
                cosine_anneal = CosineAnnealingLR(self.optimizer,
                                                  T_max=(cfg.train.num_epochs - warmup_epochs)*len(self.train_loader),
                                                  eta_min=cfg.train.optimizer.lr/10)
                self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, warmup_epochs*len(self.train_loader),
                                                        after_scheduler=cosine_anneal
                                                        )

        elif self.mode == "test":
            # if 'exp_dir' not in cfg.test:
            #     cfg.test.exp_dir = os.path.join(os.path.dirname(cfg.filename), cfg.start_time)
            # self.test_loader = self.get_dataloaders(cfg.dataset, mode=self.mode)
            self.test_loader = self.get_dataloaders(cfg.dataset, splits=['test'], dist_cfg=cfg.dist_dict)

        self.model_cast_move()

    def move_ddp_model(self):
        device = torch.device("cuda", self.cfg.dist_dict.proc_id)
        self.model = self.model.to(device)
        self.model = DDP(self.model, device_ids=[self.cfg.dist_dict.proc_id],
                         output_device=self.cfg.dist_dict.proc_id, find_unused_parameters=True)

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    def model_params(self, **kwargs):
        pass

    def get_dataloaders(self, data_cfg, splits, dist_cfg=None):
        dataloaders = []
        for split in splits:
            dataset = DATASETS.get(data_cfg.type)(data_cfg, split=split)

            batch_size = getattr(data_cfg, split).batch_size if hasattr(getattr(data_cfg, split), 'batch_size') \
                else data_cfg.batch_size
            data_params = dict(batch_size=batch_size, shuffle=(split == 'train'),
                               num_workers=data_cfg.num_workers,
                               pin_memory=True)
            if hasattr(dataset, 'collate_fn'):
                data_params['collate_fn'] = dataset.collate_fn

            # if split =='train':
                # contains_anomaly = np.array(dataset.windows_contains_anomaly)
                # batch_sampler = SamplerFactory().get(
                #     class_idxs=[np.logical_not(contains_anomaly).nonzero()[0], contains_anomaly.nonzero()[0]],
                #     batch_size=batch_size,
                #     n_batches=len(dataset) // batch_size,
                #     alpha=0.3,
                #     kind='fixed'
                # )
                #
                #
                # data_params['batch_sampler'] = batch_sampler
                # del data_params['batch_size']
                # del data_params['shuffle']

            if dist_cfg is not None:
                data_params['drop_last'] = split == 'train'
                logger.info('distributed sampler proc_id: {}'.format(dist_cfg.proc_id))
                sampler = DistributedSampler(dataset, num_replicas=dist_cfg.world_size, rank=dist_cfg.proc_id,
                                   shuffle=data_params['shuffle'], drop_last=data_params['drop_last'])
                data_params['sampler'] = sampler
                del data_params['shuffle']

            dataloader = DataLoader(dataset, **data_params)
            dataloaders.append(dataloader)

        if len(dataloaders) != 1:
            return dataloaders
        else:
            return dataloaders[0]

    @abstractmethod
    def model_forward(self, data, evaluate=False, **kwargs):
        pass

    @abstractmethod
    def calculate_score(self, epoch, epoch_dict):
        pass

    @abstractmethod
    def visualize_outputs(self, output_data, exp_dir):
        pass

    @staticmethod
    def init_epoch_dict():
        return Dict(losses=dict())

    def init_train_epoch_dict(self):
        epoch_dict = self.init_epoch_dict()
        return epoch_dict

    @abstractmethod
    def init_eval_epoch_dict(self):
        pass

    def end_of_train_epoch(self, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def create_losses(loss_cfg):
        pass

    def train_iteration(self, data, epoch_dict, epoch):
        self.optimizer.zero_grad()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        loss, report, _, loss_dict = self.model_forward(data, epoch=epoch)
        loss.backward()
        self.optimizer.step()

        if self.cfg.distributed:
            report['total_loss'] = reduce_tensor(report['total_loss'].data, world_size=self.cfg.dist_dict.world_size).item()
        else:
            report['total_loss'] = report['total_loss'].item()

        for loss_key in loss_dict:
            if loss_key not in epoch_dict.losses:
                epoch_dict.losses[loss_key] = []

            if self.cfg.distributed:
                loss_value = reduce_tensor(loss_dict[loss_key].data, world_size=self.cfg.dist_dict.world_size).item()
                epoch_dict.losses[loss_key].append(loss_value)
                loss_dict[loss_key] = loss_value
            else:
                loss_value = loss_dict[loss_key].item()
                epoch_dict.losses[loss_key].append(loss_value)
                loss_dict[loss_key] = loss_value


        report["lr"] = self.optimizer.param_groups[0]["lr"]
        report['loss_dict'] = loss_dict
        # logger.debug("iteration report:\n{}".format(report))
        # self.grad_clip_norm(epoch=epoch)
        # self.train_loader.dataset.set_temporal_scale(120)
        return report, epoch_dict

    def grad_clip_norm(self, **kwargs):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

    def eval_iteration(self, data, epoch_dict, epoch):
        with torch.no_grad():
            loss, report, clip_dicts, loss_dict = self.model_forward(data, evaluate=True, epoch=epoch)

        for loss_key in loss_dict:
            if loss_key not in epoch_dict.losses:
                epoch_dict.losses[loss_key] = []
            epoch_dict.losses[loss_key].append(loss_dict[loss_key].to('cpu'))

        epoch_dict.predictions.update(clip_dicts[-1].predictions)
        epoch_dict.targets.update(clip_dicts[-1].targets)
        return report, epoch_dict, clip_dicts

    @staticmethod
    def iter_info(report_data):
        info = "Loss: {:.3f}".format(report_data["total_loss"])
        if 'loss_dict' in report_data:
            for key, value in report_data['loss_dict'].items():
                if key == 'total_loss':
                    continue
                info += '  {}: {:.3f}'.format(key, value)
        if 'lr' in report_data:
            info += "   Lr: {:.3e}".format(report_data['lr'])
        return info

    @staticmethod
    def epoch_report(epoch_dict):
        report = dict()
        report['message'] = ""
        for loss_key in epoch_dict.losses:
            report["epoch/{}".format(loss_key)] = float(np.array(epoch_dict.losses[loss_key]).mean())
            report['message'] += " {}: {:.2f} ".format(loss_key, report["epoch/{}".format(loss_key)])
        logger.debug("\n{}".format(report))
        return report

    def save_variables(self):
        if self.cfg.distributed:
            variables_dict = Dict(optimizer=self.optimizer.state_dict(),
                                  model=self.model.module.state_dict())
        else:
            variables_dict = Dict(optimizer=self.optimizer.state_dict(),
                                  model=self.model.state_dict())
        if hasattr(self, "scheduler"):
            variables_dict.scheduler = self.scheduler.state_dict()

        return variables_dict

    def load_variables(self, variables):
        self.model.load_state_dict(variables["model"])
        # if self.mode == "train":
        #     if "optimizer" in variables:
        #         self.optimizer.load_state_dict(variables["optimizer"])
        #     if hasattr(self, "scheduler"):
        #         self.scheduler.load_state_dict(variables["scheduler"])
        self.model_cast_move()

    def model_cast_move(self):
        if torch.cuda.is_available():
            if self.cfg.dist_dict is not None:
                self.model = self.model.to('cuda:{}'.format(self.cfg.dist_dict.proc_id))
            else:
                self.model = self.model.cuda()
            logger.info("model has been moved to gpu")
        self.model = self.model.float()
