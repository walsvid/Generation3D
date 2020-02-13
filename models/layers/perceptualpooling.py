import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualPooling(nn.Module):
    def __init__(self, options):
        super(PerceptualPooling, self).__init__()
        self.options = options

    def forward(self, img_featuremaps, pc, trans_mat):
        x1, x2, x3, x4, x5 = img_featuremaps
        f1 = F.interpolate(x1, size=self.options.map_size, mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=self.options.map_size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=self.options.map_size, mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=self.options.map_size, mode='bilinear', align_corners=True)
        f5 = F.interpolate(x5, size=self.options.map_size, mode='bilinear', align_corners=True)
        # pc B*N*3
        homo_const = torch.ones((pc.shape[0], pc.shape[1], 1), device=pc.device, dtype=pc.dtype)
        homo_pc = torch.cat((pc, homo_const), dim=-1)
        pc_xyz = torch.matmul(homo_pc, trans_mat)  # pc_xyz B*N*3
        pc_xy = torch.div(pc_xyz[:, :, :2], (pc_xyz[:, :, 2:]+1e-8))  # avoid divide zero
        pc_xy = torch.clamp(pc_xy, 0.0, 136.0)  # pc_xy B*N*2

        half_resolution = (self.options.map_size - 1) / 2.
        nomalized_pc_xy = ((pc_xy - half_resolution)/half_resolution).unsqueeze(1)
        outf1 = F.grid_sample(f1, nomalized_pc_xy, align_corners=True)
        outf2 = F.grid_sample(f2, nomalized_pc_xy, align_corners=True)
        outf3 = F.grid_sample(f3, nomalized_pc_xy, align_corners=True)
        outf4 = F.grid_sample(f4, nomalized_pc_xy, align_corners=True)
        outf5 = F.grid_sample(f5, nomalized_pc_xy, align_corners=True)
        out = torch.cat((outf1, outf2, outf3, outf4, outf5), dim=1)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (Map pc to ' \
            + str(self.options.map_size) + ' x ' + str(self.options.map_size) \
            + ' plane)'
