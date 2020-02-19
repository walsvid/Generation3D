import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers import PerceptualPooling, PointMLP, SDFDecoder  # , LatentAdjust, SDFDecoder


class CIPmodel(nn.Module):
    def __init__(self, options):
        super(CIPmodel, self).__init__()
        self.options = options
        self.nn_encoder = get_backbone(self.options)
        self.percep_pooling = PerceptualPooling(self.options)

        self.point_mlp_global = PointMLP(self.options)
        self.point_mlp_local = PointMLP(self.options)

        # self.latent_adjust_global = LatentAdjust(self.options)
        # self.latent_adjust_local = LatentAdjust(self.options)

        self.nn_decoder_global = SDFDecoder(self.options, feat_channel=1024+512)
        self.nn_decoder_local = SDFDecoder(self.options, feat_channel=1472+512)

    def forward(self, img, pc, pc_rot, trans_mat):
        batch_size = img.size(0)
        img_featuremaps, img_feat_global = self.nn_encoder(img)
        img_feat_local = self.percep_pooling(img_featuremaps, pc, trans_mat)
        # change B*N*3 --> B*3*1*N
        pc_rot = pc_rot.unsqueeze(3).permute(0, 2, 3, 1)
        point_feat_global = self.point_mlp_global(pc_rot)
        point_feat_local = self.point_mlp_local(pc_rot)
        #assert point_feat_global.shape[-1] == point_feat_local.shape[-1]
        num_points = point_feat_global.shape[-1]
        expand_img_feat_global = img_feat_global.expand(-1, -1, -1, num_points)
        expand_img_feat_local = img_feat_local.expand(-1, -1, -1, num_points)
        latent_feat_global = torch.cat([point_feat_global, expand_img_feat_global], dim=1)
        latent_feat_local = torch.cat([point_feat_local, expand_img_feat_local], dim=1)

        pred_sdf_global = self.nn_decoder_global(latent_feat_global)
        pred_sdf_local = self.nn_decoder_local(latent_feat_local)
        pred_sdf = pred_sdf_global + pred_sdf_local
        if self.options.tanh:
            pred_sdf = F.tanh(pred_sdf)
        pred_sdf = pred_sdf.squeeze(2)

        # compact_feat_global = self.latent_adjust_global(latent_feat_global)
        # compact_feat_local = self.latent_adjust_local(latent_feat_local)
        # sdf_global = self.nn_decoder_global(compact_feat_global)
        # sdf_local = self.nn_decoder_local(compact_feat_local)

        # sdfs = []
        # for i in range(self.options.parts):
        #     sdfs.append(sdf_global+sdf_local)

        # sdf_each_part = torch.cat(sdfs, dim=3)

        # # prob = F.softmin(sdf_all, dim=3)
        # sdf_min, min_idx = torch.min(sdf_each_part, dim=3, keepdim=True)
        # #F.interpolate(a, size=(5,5), mode='bilinear', align_corners=True)
        return {
            "pred_sdf": pred_sdf,
            "pred_sdf_global": pred_sdf_global,
            "pred_sdf_local": pred_sdf_local
        }
