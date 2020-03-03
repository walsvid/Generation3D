import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.sdf_threshold = self.options.loss.sdf.threshold
        self.sdf_near_surface_weight = self.options.loss.sdf.weights.near_surface
        self.sdf_scale = self.options.loss.sdf.weights.scale

    def forward(self, pred_dict, input_batch):
        targets = input_batch["sdf_value"]
        outputs = pred_dict["pred_sdf"]

        weight_mask = (targets < self.sdf_threshold).float() * self.sdf_near_surface_weight \
            + (targets >= self.sdf_threshold).float()
        sdf_loss = torch.mean(torch.abs(targets * self.sdf_scale - outputs) * self.sdf_near_surface_weight)
        sdf_loss = sdf_loss * self.options.loss.sdf.coefficient
        sdf_loss_realvalue = torch.mean(torch.abs(targets - outputs / self.sdf_scale))

        gt_sign = torch.gt(targets, 0.5)
        pred_sign = torch.gt(outputs, 0.5)
        accuracy = torch.mean(torch.eq(gt_sign, pred_sign).float())

        loss = sdf_loss

        return loss, {
            "loss": loss,
            "sdf_loss": sdf_loss,
            "sdf_loss_realvalue": sdf_loss_realvalue,
            "accuracy": accuracy,
        }
