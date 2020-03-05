import torch
import torch.nn as nn


class GANBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred_dict, targets):
        outputs = pred_dict["pred_label"]

        return self.bce(outputs, targets)
