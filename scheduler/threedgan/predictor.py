import os
import numpy as np
import struct
import torch

from scheduler.base.base_predictor import Predictor
from models.zoo.threedgan import ThreeDGANmodel
from utils.vis.vis_vox import plotFromVoxels


class ThreeDGANPredictor(Predictor):
    def init_auxiliary(self):
        pass

    def init_model(self):
        return ThreeDGANmodel(self.options.model.threedgan)

    def generateZ(self, options, batch_size, device):
        if options.z_distribution == "norm":
            Z = torch.Tensor(batch_size, options.z_size).normal_(0, 0.33).to(device)
        elif options.z_distribution == "uni":
            Z = torch.randn(batch_size, options.z_size, device=device)
        else:
            raise NotImplementedError("The distribution of noise not found.")
        return Z

    def predict_step(self, input_batch):
        self.model.eval()
        with torch.no_grad():
            X = input_batch["voxel"]
            device = X.device
            batch_size = X.size(0)
            # D = self.model.module.D
            G = self.model.module.G
            Z = self.generateZ(self.options.model.threedgan, batch_size, device)
            fake = G(Z)
            result = fake
            self.save_inference_results(input_batch, result)

    def save_inference_results(self, inputs, results):
        batch_size = inputs["voxel"].size(0)
        predict_dir = self.options.predict_dir
        for b in range(batch_size):
            voxel = results['pred_voxel'][b].squeeze().cpu().numpy()
            obj_nm = inputs['filename'][b]
            voxel_plot_file = os.path.join(predict_dir, obj_nm + ".png")
            plotFromVoxels(voxel, voxel_plot_file)
        # marching_cube_command = "./external/isosurface/computeMarchingCubes"
        # batch_size = inputs["images"].shape[0]
        # iso = 0.003
        # predict_dir = self.options.predict_dir
        # for b in range(batch_size):
        #     pred_sdf_val = np.swapaxes(results[b], 0, 1)  # N,1
        #     obj_nm = inputs['filename'][b]
        #     if not obj_nm.endswith('_0'):
        #         continue
        #     sdf_params = inputs['sdf_params'][b].cpu().numpy()
        #     cube_obj_file = os.path.join(predict_dir, obj_nm + ".obj")
        #     sdf_file = os.path.join(predict_dir, obj_nm + ".dist")
        #     self.to_binary((self.options.model.disn.resolution - 1), sdf_params, pred_sdf_val, sdf_file)
        #     command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(iso)
        #     os.system(command_str)
        #     command_str2 = "rm -rf " + sdf_file
        #     os.system(command_str2)
