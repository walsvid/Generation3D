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
    def __init__(self, file_root, file_list_name, img_dir, sdf_dir, normalization, shapenet_options, logger):
        super(ShapeNet, self).__init__()
        self.file_root = file_root
        self.sdf_dir = os.path.join(self.file_root, sdf_dir)
        self.img_dir = os.path.join(self.file_root, img_dir)
        print(self.sdf_dir)
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

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_original' in h5_f.keys()
                    and 'pc_sdf_sample' in h5_f.keys()
                    and 'norm_params' in h5_f.keys()):
                ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
                sample_sdf = h5_f['pc_sdf_sample'][:-10000].astype(np.float32)
                ori_pt = ori_sdf[:, :3]  # , ori_sdf[:,3]
                ori_sdf_val = None
                if sample_sdf.shape[1] == 4:
                    sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3:]
                else:
                    sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
                norm_params = h5_f['norm_params'][:]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return sample_pt, sample_sdf_val, norm_params, sdf_params

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5" % num)
        cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat = None, None, None, None, None
        with h5py.File(img_h5, 'r') as h5_f:
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)
            obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
            regress_mat = h5_f["regress_mat"][:].astype(np.float32)
            cam_mat = h5_f["K"][:].astype(np.float32)
            cam_pos = h5_f["RT"][:].astype(np.float32)
            if self.shapenet_options.img_alpha:
                img_arr = h5_f["img_arr"][:].astype(np.float32)
                img_arr = img_arr / 255.
            else:
                img_raw = h5_f["img_arr"][:]
                img_arr = img_raw[:, :, :3]
                if self.shapenet_options.augcolorfore or self.shapenet_options.augcolorback:
                    r_aug = 60 * np.random.rand() - 30
                    g_aug = 60 * np.random.rand() - 30
                    b_aug = 60 * np.random.rand() - 30
                if self.shapenet_options.augcolorfore:
                    img_arr[img_raw[:, :, 3] != 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] != 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] != 0, 2] + b_aug
                if self.shapenet_options.augcolorback:
                    img_arr[img_raw[:, :, 3] == 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] == 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] == 0, 2] + b_aug
                else:
                    img_arr[img_raw[:, :, 3] == 0] = [255, 255, 255]
                img_arr = np.clip(img_arr, 0, 255)
                img_arr = img_arr.astype(np.float32) / 255.
            if self.shapenet_options.resize_with_constant_border:
                img = transform.resize(img_arr, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False).astype(np.float32)  # to match behavior of old versions
            else:
                img = transform.resize(img_arr, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32)
        return img, cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat

    def __getitem__(self, index):
        cat_id, obj, num = self.file_names[index]
        # read sdf
        sdf_file = os.path.join(self.sdf_dir, cat_id, obj, "ori_sample.h5")
        sample_pt, sample_sdf_val, norm_params, sdf_params = self.get_sdf_h5(sdf_file, cat_id, obj)
        sample_sdf_val = sample_sdf_val - 0.003  # the value of iso-surface 0.003
        sample_sdf_val = sample_sdf_val.T
        choice = np.asarray(random.sample(range(sample_pt.shape[0]), self.shapenet_options.num_points), dtype=np.int32)
        # read image
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        img, cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat = self.get_img(img_dir, num)
        # to tensor
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img
        sdf_points = torch.from_numpy(sample_pt[choice, :])
        sdf_value = torch.from_numpy(sample_sdf_val[:, choice])
        norm_params = torch.from_numpy(norm_params)
        sdf_params = torch.from_numpy(sdf_params)
        regress_mat = torch.from_numpy(regress_mat)
        trans_mat = torch.from_numpy(trans_mat)
        filename = '_'.join(map(lambda x: str(x), [cat_id, obj, num]))

        if self.shapenet_options.rot:
            sample_pt_rot = np.dot(sample_pt[choice, :], obj_rot_mat)
        else:
            sample_pt_rot = sample_pt[choice, :]
        sdf_points_rot = torch.from_numpy(sample_pt_rot)

        return {
            "images": img_normalized,
            "images_orig": img,
            "sdf_points": sdf_points,
            "sdf_points_rot": sdf_points_rot,
            "sdf_value": sdf_value,
            "norm_params": norm_params,
            "sdf_params": sdf_params,
            "trans_mat": trans_mat,
            "regress_mat": regress_mat,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)
