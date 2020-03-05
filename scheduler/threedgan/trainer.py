from scheduler.base.base_trainer import Trainer

from models.zoo.threedgan import ThreeDGANmodel
from models.losses.gan_loss import GANBCELoss
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach

import torch
import torch.nn as nn


class ThreeDGANTrainer(Trainer):
    def init_auxiliary(self):
        pass

    def init_model(self):
        return ThreeDGANmodel(self.options.model.threedgan)

    def init_loss_functions(self):
        return GANBCELoss().cuda()

    def train_step(self, input_batch):
        X = input_batch["voxel"]
        device = X.device
        batch_size = X.size(0)
        D = self.model.module.D
        G = self.model.module.G
        D_solver = self.optimizer["optimizer_d"]
        G_solver = self.optimizer["optimizer_g"]

        Z = self.generateZ(self.options.model.threedgan, batch_size, device)
        if self.options.model.threedgan.soft_label:
            real_labels = torch.Tensor(batch_size).uniform_(0.7, 1.2).cuda()
            fake_labels = torch.Tensor(batch_size).uniform_(0, 0.3).cuda()
        else:
            real_labels = torch.ones(batch_size).cuda()
            fake_labels = torch.zeros(batch_size).cuda()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        d_real = D(X)
        # import ipdb
        # ipdb.set_trace()
        d_real_loss = self.criterion(d_real, real_labels)

        fake = G(Z)
        d_fake = D(fake['pred_voxel'])
        d_fake_loss = self.criterion(d_fake, fake_labels)

        d_loss = d_real_loss + d_fake_loss

        d_real_acu = torch.ge(d_real['pred_label'].squeeze(), 0.5).float()
        d_fake_acu = torch.le(d_fake['pred_label'].squeeze(), 0.5).float()
        d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

        if d_total_acu <= self.options.model.threedgan.d_thresh:
            D.zero_grad()
            d_loss.backward()
            D_solver.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Z = self.generateZ(self.options.model.threedgan, batch_size, device)

        fake = G(Z)
        d_fake = D(fake['pred_voxel'])
        g_loss = self.criterion(d_fake, real_labels)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        G_solver.step()

        out = fake
        loss_summary = {
            "loss_D_R": d_real_loss,
            "loss_D_F": d_fake_loss,
            "loss_D": d_loss,
            "loss_G": g_loss,
            "acc_D": d_total_acu
        }

        self.losses.update(g_loss.detach().cpu().item())
        # # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def generateZ(self, options, batch_size, device):
        if options.z_distribution == "norm":
            Z = torch.Tensor(batch_size, options.z_size).normal_(0, 0.33).to(device)
        elif options.z_distribution == "uni":
            Z = torch.randn(batch_size, options.z_size, device=device)
        else:
            raise NotImplementedError("The distribution of noise not found.")
        return Z

    def optimizers_dict(self):
        return {'optimizer_d': self.optimizer['optimizer_d'],
                'optimizer_g': self.optimizer['optimizer_g'],
                'lr_scheduler': self.lr_scheduler}

    def log_step(self, loss_summary):
        self.logger.info("Epoch %03d/%03d, Step %06d/%06d | %06d/%06d, Time elapsed %s, DLoss %.5f, GLoss %.5f, Acc %.3f" % (
            self.epoch_count, self.options.train.num_epochs,
            self.step_count - ((self.epoch_count - 1) * self.dataset_size // (
                self.options.train.summary_steps * self.options.train.summary_steps
            )), self.dataset_size,
            self.step_count, self.options.train.num_epochs * self.dataset_size,
            self.time_elapsed,
            loss_summary["loss_D"], loss_summary["loss_G"],
            loss_summary["acc_D"] * 100))
