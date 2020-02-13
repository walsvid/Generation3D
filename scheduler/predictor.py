import os
import random
from logging import Logger

import imageio
import struct
import numpy as np
import torch

from torch.utils.data import DataLoader
from scheduler.base import CheckpointRunner
from models.p2m import P2MModel
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer

RESOLUTION = 256+1


class Predictor(CheckpointRunner):
    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.gpu_inference = self.options.num_gpus > 0
        if self.gpu_inference == 0:
            raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                      "might be fixed in the future.")
        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create ellipsoid
                self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

    def models_dict(self):
        return {'model': self.model}

    def predict_step(self, input_batch):
        self.model.eval()
        # Run inference
        with torch.no_grad():
            images = input_batch["images"]
            out = self.model(images)
            self.save_inference_results(input_batch, out, None)

    def predict(self):
        self.logger.info("Running predictions...")
        predict_data_loader = DataLoader(self.dataset,
                                         batch_size=self.options.test.batch_size,
                                         pin_memory=self.options.pin_memory,
                                         collate_fn=self.dataset_collate_fn)

        for step, batch in enumerate(predict_data_loader):
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            if self.gpu_inference:
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                          "might be fixed in the future.")
            self.predict_step(batch)

    def save_inference_results(self, inputs, outputs, results):
        if self.options.model.name == "pixel2mesh":
            batch_size = inputs["images"].size(0)
            for i in range(batch_size):
                basename = os.path.join(self.options.predict_dir, inputs["filename"][i])
                mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
                verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
                for k, vert in enumerate(verts):
                    meshname = basename + "_%d.obj" % (k + 1)
                    vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                    mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                    np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")
                # TODO: Render GIF
