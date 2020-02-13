import os
import pickle

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix

import utils.config as config


def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)


"""
dat = [info['coords'],
       info['support']['stage1'],
       info['support']['stage2'],
       info['support']['stage3'],
       info['support']['stage4'],
       [info['unpool_idx']['stage1_2'],
        info['unpool_idx']['stage2_3'],
        info['unpool_idx']['stage3_4']
       ],
       [np.zeros((1,4), dtype=np.int32)]*4,
       [np.zeros((1,4))]*4,
       [info['lap_idx']['stage1'],
        info['lap_idx']['stage2'],
        info['lap_idx']['stage3'],
        info['lap_idx']['stage4']
       ],
      ]
"""


class Ellipsoid(object):
    def __init__(self, mesh_pos, ellip_file=config.ELLIPSOID_PATH):
        with open(ellip_file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        # shape: n_pts * 3
        self.coord = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos, dtype=torch.float)

        # edges & faces & lap_idx
        # edge: num_edges * 2
        # faces: num_faces * 4
        # laplace_idx: num_pts * 10
        self.edges, self.laplace_idx = [], []

        for i in range(4):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            self.laplace_idx.append(torch.tensor(fp_info[8][i], dtype=torch.long))

        # unpool index, 1-2 2-3 3-4
        # num_pool_edges * 2
        # pool_01: 462 * 2, pool_12: 1848 * 2
        self.unpool_idx = [torch.tensor(fp_info[5][i], dtype=torch.long) for i in range(3)]

        # loops and adjacent edges
        self.adj_mat = []
        for i in range(1, 5):
            # 0: np.array, 2D, pos
            # 1: np.array, 1D, vals
            # 2: tuple - shape, n * n
            adj_mat = torch_sparse_tensor(*fp_info[i][1])
            self.adj_mat.append(adj_mat)

        ellipsoid_dir = os.path.dirname(ellip_file)
        self.faces = []
        self.obj_fmt_faces = []
        # faces: f * 3, original ellipsoid, and two after deformations
        for i in range(1, 5):
            face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
            faces = np.loadtxt(face_file, dtype='|S32')
            self.obj_fmt_faces.append(faces)
            self.faces.append(torch.tensor(faces[:, 1:].astype(np.int) - 1))
