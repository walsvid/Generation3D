import json
import os
import pickle

import numpy as np
import torch
import h5py
import random
from PIL import Image
from skimage import io, transform
# from torch.utils.data.dataloader import default_collate

import utils.config as config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    def __init__(self, file_root, file_list_name, img_dir, mesh_pos, normalization, shapenet_options, logger):
        super(ShapeNet, self).__init__()
        self.file_root = file_root
        self.mesh_pos = mesh_pos
        # self.sdf_dir = os.path.join(self.file_root, sdf_dir)
        self.img_dir = os.path.join(self.file_root, img_dir)
        # print(self.sdf_dir)
        print(self.img_dir)
        self.file_names = []
        for lst in file_list_name:
            category_name = lst.split("_")[0]
            with open(os.path.join(self.file_root, "meta", lst), "r") as fp:
                for l in fp:
                    for r in range(24):
                        self.file_names.append((category_name, l.strip(), r))
        self.normalization = normalization
        self.shapenet_options = shapenet_options
        self.logger = logger

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5" % num)
        with h5py.File(img_h5, 'r') as h5_f:
            img_ori = h5_f["img_ori"][:].astype(np.float32) / 255.
            # img_x1 = h5_f["img_x1"][:].astype(np.float32) / 255.
            # img_x2 = h5_f["img_x2"][:].astype(np.float32) / 255.
            # img_x4 = h5_f["img_x4"][:].astype(np.float32) / 255.
            # img_x8 = h5_f["img_x8"][:].astype(np.float32) / 255.
            # img_x1 = transform.rescale(img_x1, 8, mode='constant', anti_aliasing=False, multichannel=True).astype(np.float32)
            # img_x2 = transform.rescale(img_x2, 4, mode='constant', anti_aliasing=False, multichannel=True).astype(np.float32)
            # img_x4 = transform.rescale(img_x4, 2, mode='constant', anti_aliasing=False, multichannel=True).astype(np.float32)

            points = h5_f["ptnm"][:, :3].astype(np.float32)
            normals = h5_f["ptnm"][:, 3:].astype(np.float32)
        points -= np.array(self.mesh_pos)
        assert points.shape[0] == normals.shape[0]
        return img_ori, points, normals

    def __getitem__(self, index):
        cat_id, obj, num = self.file_names[index]
        # read images pyramid
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        img_ori, pt, nm = self.get_img(img_dir, num)
        # to tensor
        img_ori = torch.from_numpy(np.transpose(img_ori, (2, 0, 1)))
        img_ori_normalized = self.normalize_img(img_ori) if self.normalization else img_ori

        # imgx1 = torch.from_numpy(np.transpose(imgx1, (2, 0, 1)))
        # imgx1_normalized = self.normalize_img(imgx1) if self.normalization else imgx1

        # imgx2 = torch.from_numpy(np.transpose(imgx2, (2, 0, 1)))
        # imgx2_normalized = self.normalize_img(imgx2) if self.normalization else imgx2

        # imgx4 = torch.from_numpy(np.transpose(imgx4, (2, 0, 1)))
        # imgx4_normalized = self.normalize_img(imgx4) if self.normalization else imgx4

        # imgx8 = torch.from_numpy(np.transpose(imgx8, (2, 0, 1)))
        # imgx8_normalized = self.normalize_img(imgx8) if self.normalization else imgx8

        points = torch.from_numpy(pt)
        normals = torch.from_numpy(nm)
        filename = '_'.join(map(lambda x: str(x), [cat_id, obj, num]))

        return {
            "images": img_ori_normalized,
            "images_ori": img_ori,
            "points": points,
            "normals": normals,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)
