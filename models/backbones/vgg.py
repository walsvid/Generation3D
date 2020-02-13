import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import utils.config as config


class VGG16(nn.Module):
    def __init__(self, num_classes=1024, pretrained=False):
        super(VGG16, self).__init__()
        self.features_dim = 1472    # 64 + 128 + 256 + 512 + 512
        p = [0, 6, 13, 23, 33, 43]
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        self.block5 = torch.nn.Sequential()
        vgg = models.vgg16_bn(pretrained=pretrained)
        for x in range(p[0], p[1]):
            self.block1.add_module(str(x), vgg.features[x])
        for x in range(p[1], p[2]):
            self.block2.add_module(str(x), vgg.features[x])
        for x in range(p[2], p[3]):
            self.block3.add_module(str(x), vgg.features[x])
        for x in range(p[3], p[4]):
            self.block4.add_module(str(x), vgg.features[x])
        for x in range(p[4], p[5]):
            self.block5.add_module(str(x), vgg.features[x])

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1, padding=0)
        )
        self._initialize_weights(self.block6)

    def _initialize_weights(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        perceptual_feat = [x1, x2, x3, x4, x5]
        global_feat = x6
        return perceptual_feat, global_feat
