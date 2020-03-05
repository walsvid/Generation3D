import json
import os
import pickle

import numpy as np
import torch
import h5py
import random
from PIL import Image
from skimage import io, transform

import utils.config as config
from datasets.base_dataset import BaseDataset
from utils.binvox_rw import read_as_3d_array


class ThreeDGANShapeNet(BaseDataset):
    def __init__(self, file_root, file_list_name, voxel_dir, normalization, shapenet_options, logger):
        super(ThreeDGANShapeNet, self).__init__()
        self.file_root = file_root
        self.voxel_dir = os.path.join(self.file_root, voxel_dir)
        print(self.voxel_dir)
        self.file_names = []
        for lst in file_list_name:
            category_name = lst.split("_")[0]
            with open(os.path.join(self.file_root, "meta", lst), "r") as fp:
                for l in fp:
                    self.file_names.append((category_name, l.strip()))
        # self.normalization = normalization
        # self.shapenet_options = shapenet_options
        self.logger = logger

    def get_binvox(self, binvox_file, cat_id, obj):
        with open(binvox_file, 'rb') as fp:
            voxel = read_as_3d_array(fp)

        return voxel.data.astype(np.float32)

    def __getitem__(self, index):
        cat_id, obj = self.file_names[index]
        # read binvox
        binvox_file = os.path.join(self.voxel_dir, cat_id, obj, "model.binvox")
        binvox_np = self.get_binvox(binvox_file, cat_id, obj)
        filename = '_'.join(map(lambda x: str(x), [cat_id, obj]))

        voxel = torch.from_numpy(binvox_np)

        return {
            "voxel": voxel,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)
