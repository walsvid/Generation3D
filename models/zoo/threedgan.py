from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeDGANmodel(nn.Module):
    def __init__(self, options):
        super(ThreeDGANmodel, self).__init__()
        self.options = options

        self.G = _G(self.options)
        self.D = _D(self.options)


class _G(torch.nn.Module):
    def __init__(self, options):
        super(_G, self).__init__()
        self.z_size = 200       # options.z_size
        self.bias = False       # options.bias
        self.voxel_size = 32    # options.voxel_size

        if self.voxel_size == 32:
            padd = (1, 1, 1)
        else:
            padd = (0, 0, 0)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.z_size, self.voxel_size * 8, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.BatchNorm3d(self.voxel_size * 8),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 8, self.voxel_size * 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 4),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 4, self.voxel_size * 2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 2),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 2, self.voxel_size, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size),
            torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid())

    def forward(self, z):
        out = z.view(-1, self.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return {
            "pred_voxel": out
        }


class _D(torch.nn.Module):
    def __init__(self, options):
        super(_D, self).__init__()
        self.z_size = 200       # options.z_size
        self.bias = False       # options.bias
        self.voxel_size = 32    # options.voxel_size
        self.leak_value = 0.2   # options.leak_value

        if self.voxel_size == 32:
            padd = (1, 1, 1)
        else:
            padd = (0, 0, 0)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.voxel_size, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size),
            torch.nn.LeakyReLU(self.leak_value))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.voxel_size, self.voxel_size * 2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 2),
            torch.nn.LeakyReLU(self.leak_value))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.voxel_size * 2, self.voxel_size * 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 4),
            torch.nn.LeakyReLU(self.leak_value))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.voxel_size * 4, self.voxel_size * 8, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 8),
            torch.nn.LeakyReLU(self.leak_value))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.voxel_size * 8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid())

    def forward(self, x):
        out = x.view(-1, 1, self.voxel_size, self.voxel_size, self.voxel_size)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1)
        return {
            "pred_label": out
        }
