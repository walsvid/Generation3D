import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers import PerceptualPooling, PointMLP, SDFDecoder


class DISNmodel(nn.Module):
    def __init__(self, options):
        super(DISNmodel, self).__init__()
        self.options = options
        self.nn_encoder = get_backbone(self.options)
        self.percep_pooling = PerceptualPooling(self.options)

        self.point_mlp_global = PointMLP(self.options)
        self.point_mlp_local = PointMLP(self.options)

        self.nn_decoder_global = SDFDecoder(self.options, feat_channel=1024 + 512)
        self.nn_decoder_local = SDFDecoder(self.options, feat_channel=1472 + 512)

    def forward(self, input_batch):
        img = input_batch["images"]
        pc = input_batch["sdf_points"]
        pc_rot = input_batch["sdf_points_rot"]
        trans_mat = input_batch["trans_mat"]

        batch_size = img.size(0)
        img_featuremaps, img_feat_global = self.nn_encoder(img)
        img_feat_local = self.percep_pooling(img_featuremaps, pc, trans_mat)
        # change B*N*3 --> B*3*1*N
        pc_rot = pc_rot.unsqueeze(3).permute(0, 2, 3, 1)
        point_feat_global = self.point_mlp_global(pc_rot)
        point_feat_local = self.point_mlp_local(pc_rot)

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

        return {
            "pred_sdf": pred_sdf,
            "pred_sdf_global": pred_sdf_global,
            "pred_sdf_local": pred_sdf_local
        }
