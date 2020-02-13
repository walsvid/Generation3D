import time
from datetime import timedelta
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scheduler.base import CheckpointRunner
# from scheduler.evaluator import Evaluator

from models.losses.p2m_loss import P2MLoss
from models.p2m import P2MModel

from utils.average_meter import AverageMeter
from utils.tensor import recursive_detach
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer


class Trainer(CheckpointRunner):
    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        if self.options.model.name == "pixel2mesh":
            # Visualization renderer
            self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                         self.options.dataset.mesh_pos)
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        else:
            self.renderer = None
        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        if self.options.optim.lr_scheduler == "multistep":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
            )
        elif self.options.optim.lr_scheduler == "exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.options.optim.lr_gamma)
        # elif self.options.optim.lr_scheduler == "lambda":
        #     def lambdafunc(epoch): return 0.95 ** epoch
        #     self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafunc)
        else:
            self.lr_scheduler = None
        # Create loss functions
        if self.options.model.name == "pixel2mesh":
            self.criterion = P2MLoss(self.options.loss, self.ellipsoid).cuda()
        else:
            raise NotImplementedError("Your loss is not found")
        # Create AverageMeters for losses
        self.losses = AverageMeter()
        # Evaluators
        #self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        """
        return {
            "images": img_ori_normalized,
            "points": points,
            "normals": normals,
            "filename": filename
        }
        """
        self.model.train()
        # Grab data from the batch
        images = input_batch['images']
        # predict with model
        out = self.model(images)
        # compute loss
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())
        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        self.logger.info("Start Trainning.")
        # Create data loader at very begining
        train_data_loader = DataLoader(self.dataset,
                                       batch_size=self.options.train.batch_size * self.options.num_gpus,
                                       num_workers=self.options.num_workers,
                                       pin_memory=self.options.pin_memory,
                                       shuffle=self.options.train.shuffle)
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
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def init_with_pretrained_backbone(self):
        checkpoint_file = os.path.abspath(self.options.train.backbone_pretrained_model)
        pretrained_dict = torch.load(checkpoint_file)
        self.model.module.load_state_dict(pretrained_dict, strict=False)
        self.logger.info("Init with pre-trained backbone from %s." % checkpoint_file)

    # def test(self):
    #     for evaluator in self.evaluators:
    #         evaluator.evaluate()
