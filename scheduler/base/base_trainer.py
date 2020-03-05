from utils.tensor import recursive_detach
from utils.average_meter import AverageMeter
# from scheduler.base.evaluator import Evaluator
from scheduler.base.runner import CheckpointRunner
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import os
import torch
import torch.nn as nn

from utils.parallel import DataParallelModel, DataParallelCriterion


class Trainer(CheckpointRunner):
    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # Create auxiliary models
        self.init_auxiliary()
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = self.init_model()
            self.model = DataParallelModel(self.model.cuda(), device_ids=self.gpus)
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
        # Setup a joint optimizer for the 2 models
        self.optimizer = self.init_optimizer(self.options.optim.name)
        self.lr_scheduler = self.init_lr(self.options.optim.lr_scheduler)
        # Create loss functions
        self.criterion = self.init_loss_functions()
        self.criterion = DataParallelCriterion(self.criterion.cuda(), device_ids=self.gpus)
        # Create AverageMeters for losses
        self.losses = AverageMeter()
        # Evaluators
        # self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]
        self.dataset_size = None

    def init_auxiliary(self):
        pass

    def init_model(self):
        raise NotImplementedError("Your model is not found")

    def init_loss_functions(self):
        raise NotImplementedError("Your loss is not found")

    def init_optimizer(self, optim_name):
        if optim_name == "adam":
            optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        elif optim_name == "adam_gan":
            optimizer_d = torch.optim.Adam(
                params=list(self.model.module.D.parameters()),
                lr=self.options.optim.lr_d,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=0
            )
            optimizer_g = torch.optim.Adam(
                params=list(self.model.module.G.parameters()),
                lr=self.options.optim.lr_g,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=0
            )
            return {"optimizer_d": optimizer_d, "optimizer_g": optimizer_g}
        else:
            raise NotImplementedError("Your optimizer is not found")
        return optimizer

    def init_lr(self, lr_scheduler_name):
        if lr_scheduler_name == "multistep":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor)
        elif lr_scheduler_name == "exp":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.options.optim.lr_gamma)
        elif lr_scheduler_name == "multistep_gan":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer["optimizer_d"], self.options.optim.lr_step, self.options.optim.lr_factor)
        else:
            r_scheduler = None

        return lr_scheduler

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        # Grab data from the batch, predict with model
        out = self.model(input_batch)
        # compute loss
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())
        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def get_dataloader(self):
        data_loader = DataLoader(self.dataset,
                                 batch_size=self.options.train.batch_size * self.options.num_gpus,
                                 num_workers=self.options.num_workers,
                                 pin_memory=self.options.pin_memory,
                                 shuffle=self.options.train.shuffle)
        return data_loader

    def train(self):
        self.logger.info("Start Trainning.")
        # Create data loader at very begining
        train_data_loader = self.get_dataloader()
        self.dataset_size = len(train_data_loader)

        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1
            # Reset loss
            self.losses.reset()
            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Run training step
                out = self.train_step(batch)
                self.step_count += 1
                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)
                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()
            self.dump_checkpoint()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # Debug info for filenames
        self.logger.debug(input_batch["filename"])
        # Save results in Tensorboard
        self.tensorboard_step(loss_summary)
        # Save results to log
        self.log_step(loss_summary)

    def log_step(self, loss_summary):
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def tensorboard_step(self, loss_summary):
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

    def init_with_pretrained_backbone(self):
        checkpoint_file = os.path.abspath(self.options.train.backbone_pretrained_model)
        pretrained_dict = torch.load(checkpoint_file)
        self.model.module.load_state_dict(pretrained_dict, strict=False)
        self.logger.info("Init with pre-trained backbone from %s." % checkpoint_file)

    def test(self):
        self.model.eval()
        for evaluator in self.evaluators:
            evaluator.evaluate()
        self.model.train()
