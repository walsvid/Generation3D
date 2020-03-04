from scheduler.base.base_predictor import Predictor

from models.zoo.disn import DISNmodel
import os
import numpy as np
import struct
import torch


class DISNPredictor(Predictor):
    def init_auxiliary(self):
        os.environ["LD_LIBRARY_PATH"] += os.pathsep + os.getcwd() + "/external/isosurface"

    def init_model(self):
        return DISNmodel(self.options.model)

    def predict_step(self, input_batch):
        self.model.eval()
        filenames = input_batch["filename"]
        if not filenames[0].endswith('_0'):
            return

        total_points = self.options.model.disn.resolution ** 3
        split_size = int(np.ceil(total_points / self.options.model.disn.split_chunk))
        num_sample_points = int(np.ceil(total_points / split_size))
        batch_size = input_batch["images"].shape[0]
        extra_pts = np.zeros((1, split_size * num_sample_points - total_points, 3), dtype=np.float32)
        batch_points = np.zeros((split_size, 0, num_sample_points, 3), dtype=np.float32)
        for b in range(batch_size):
            sdf_params = input_batch['sdf_params'][b].cpu()
            x_ = np.linspace(sdf_params[0], sdf_params[3], num=self.options.model.disn.resolution)
            y_ = np.linspace(sdf_params[1], sdf_params[4], num=self.options.model.disn.resolution)
            z_ = np.linspace(sdf_params[2], sdf_params[5], num=self.options.model.disn.resolution)
            z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
            x = np.expand_dims(x, 3)
            y = np.expand_dims(y, 3)
            z = np.expand_dims(z, 3)
            all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
            all_pts = all_pts.reshape(1, -1, 3)
            all_pts = np.concatenate((all_pts, extra_pts), axis=1).reshape(split_size, 1, -1, 3)
            self.logger.info('all_pts: {}'.format(all_pts.shape))
            batch_points = np.concatenate((batch_points, all_pts), axis=1)

        pred_sdf_val_all = np.zeros((split_size, batch_size, 1, num_sample_points))

        with torch.no_grad():
            for sp in range(split_size):
                images = input_batch["images"]
                trans_mat = input_batch["trans_mat"]

                pc = torch.tensor(batch_points[sp, ...].reshape(batch_size, -1, 3), device=images.device)
                pc_rot = torch.tensor(batch_points[sp, ...].reshape(batch_size, -1, 3), device=images.device)
                input_batch["sdf_points"] = pc
                input_batch["sdf_points_rot"] = pc_rot
                out = self.model(input_batch)
                pred_sdf_val = out["pred_sdf"]
                pred_sdf_val_all[sp, :, :, :] = pred_sdf_val.cpu().numpy()

        pred_sdf_val_all = np.swapaxes(pred_sdf_val_all, 0, 1)  # B, S, C=1, NUM SAMPLE
        pred_sdf_val_all = pred_sdf_val_all.reshape((batch_size, 1, -1))[:, :, :total_points]
        result = pred_sdf_val_all / self.options.loss.sdf.weights.scale

        self.save_inference_results(input_batch, result)

    def save_inference_results(self, inputs, results):
        marching_cube_command = "./external/isosurface/computeMarchingCubes"
        batch_size = inputs["images"].shape[0]
        iso = 0.003
        predict_dir = self.options.predict_dir
        for b in range(batch_size):
            pred_sdf_val = np.swapaxes(results[b], 0, 1)  # N,1
            obj_nm = inputs['filename'][b]
            if not obj_nm.endswith('_0'):
                continue
            sdf_params = inputs['sdf_params'][b].cpu().numpy()
            cube_obj_file = os.path.join(predict_dir, obj_nm + ".obj")
            sdf_file = os.path.join(predict_dir, obj_nm + ".dist")
            self.to_binary((self.options.model.disn.resolution - 1), sdf_params, pred_sdf_val, sdf_file)
            command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(iso)
            os.system(command_str)
            command_str2 = "rm -rf " + sdf_file
            os.system(command_str2)

    def to_binary(self, res, pos, pred_sdf_val_all, sdf_file):
        f_sdf_bin = open(sdf_file, 'wb')
        f_sdf_bin.write(struct.pack('i', -res))  # write an int
        f_sdf_bin.write(struct.pack('i', res))  # write an int
        f_sdf_bin.write(struct.pack('i', res))  # write an int

        positions = struct.pack('d' * len(pos), *pos)
        f_sdf_bin.write(positions)
        val = struct.pack('=%sf' % pred_sdf_val_all.shape[0], *(pred_sdf_val_all))
        f_sdf_bin.write(val)
        f_sdf_bin.close()
