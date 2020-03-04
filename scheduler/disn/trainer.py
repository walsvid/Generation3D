from scheduler.base.base_trainer import Trainer

from models.zoo.disn import DISNmodel
from models.losses.sdf_loss import SDFLoss


class DISNTrainer(Trainer):

    def init_model(self):
        return DISNmodel(self.options.model)

    def init_loss_functions(self):
        return SDFLoss(self.options).cuda()

    def log_step(self, loss_summary):
        self.logger.info("Epoch %03d/%03d, Step %06d/%06d | %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f), Realvalue %.5f, Acc %.3f" % (
            self.epoch_count, self.options.train.num_epochs,
            self.step_count - ((self.epoch_count - 1) * self.dataset_size), self.dataset_size,
            self.step_count, self.options.train.num_epochs * self.dataset_size,
            self.time_elapsed, self.losses.val, self.losses.avg,
            loss_summary["sdf_loss_realvalue"], loss_summary["accuracy"] * 100))
