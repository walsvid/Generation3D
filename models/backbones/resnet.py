import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import utils.config as config


class ResNet18(nn.Module):

    def __init__(self, num_classes=1024, pretrained=False, *args, **kwargs):
        super(ResNet18, self).__init__(*args, **kwargs)
        self.features_dim = 64 + 64 + 128 + 256 + 512
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        self.block5 = torch.nn.Sequential()
        net = models.resnet18(pretrained=pretrained)
        p = [0, 3, 5, 6, 7, 8]
        all_layers = list(net.children())
        for i in range(p[0], p[1]):
            self.block1.add_module(str(i), all_layers[i])
        for i in range(p[1], p[2]):
            self.block2.add_module(str(i), all_layers[i])
        for i in range(p[2], p[3]):
            self.block3.add_module(str(i), all_layers[i])
        for i in range(p[3], p[4]):
            self.block4.add_module(str(i), all_layers[i])
        for i in range(p[4], p[5]):
            self.block5.add_module(str(i), all_layers[i])
    #     self.block6 = nn.Sequential(
    #         nn.Conv2d(512, 1024, kernel_size=3, padding=0),
    #         nn.BatchNorm2d(1024),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
    #         nn.BatchNorm2d(1024),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(1024, num_classes, kernel_size=3, padding=0)
    #     )
    #     self._initialize_weights(self.block6)

    # def _initialize_weights(self, x):
    #     for m in x.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        # x6 = self.block6(x5)

        perceptual_feat = [x1, x2, x3, x4, x5]
        # for i in range(6):
        #     print(perceptual_feat[i].shape)
        # global_feat = x6
        return perceptual_feat
