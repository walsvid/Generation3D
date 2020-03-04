import h5py
import numpy as np
import glob
import os
import skimage
import skimage.io
import cv2
import multiprocessing as mp

error_file = 'cteate_02691156.log'
sdf_dir = '/workspace/Users/wc/shapenet/SDF_v1'
mc_dir = '/workspace/Users/wc/shapenet/march_cube_objs_v1'
img_dir = '/workspace/Users/wc/shapenet/ShapeNetRendering'
target_dir = '/workspace/Users/wc/shapenet/ssd1/P2M_ptnm_v1'

category = '02691156'

files = glob.glob('{}/{}/*/ori_sample.h5'.format(sdf_dir, category))

print('==> file found.')
# files = files[:1]


def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    return cam_mat, cam_pos


def gen(item):
    # print(item)
    *path, cate, ids, basename = item.split('/')
    h5_f = h5py.File(item)
    norm_para = h5_f['norm_params'][:]
    mc_f = item.replace('SDF_v1', 'march_cube_objs_v1')
    surf_xyz = np.loadtxt(mc_f.replace('ori_sample.h5', 'surf.xyz'))
    selec_idx = np.loadtxt(mc_f.replace('ori_sample.h5', 'sampled_idx.txt')).astype(int)
    all_ptnm = np.loadtxt(mc_f.replace('ori_sample.h5', 'isosurf.xyz'))
    vori = surf_xyz * norm_para[-1] + norm_para[:3]
    vert = np.hstack([vori, all_ptnm[selec_idx, 3:]])
    position = vert[:, :3] * 0.57
    normal = vert[:, 3:]
    h5_f.close()

    view_path = os.path.join(img_dir, cate, ids, 'rendering', 'rendering_metadata.txt')
    target_filedir = os.path.join(target_dir, cate, ids)
    cam_params = np.loadtxt(view_path)
    # print('{}/{}'.format(cate, ids))
    for index, param in enumerate(cam_params):
        # camera tranform
        strview = str(index).zfill(2)
        x1_img_path = os.path.join(target_filedir, '{}_x1.png'.format(strview))

        if not os.path.exists(x1_img_path):
            print('===> not exist {}'.format(ids))
            break
        x1 = skimage.io.imread(x1_img_path)
        x2 = skimage.io.imread(x1_img_path.replace('_x1.png', '_x2.png'))
        x4 = skimage.io.imread(x1_img_path.replace('_x1.png', '_x4.png'))
        x8 = skimage.io.imread(x1_img_path.replace('_x1.png', '_x8.png'))
        xori = skimage.io.imread(x1_img_path.replace('_x1.png', '.png'))

        cam_mat, cam_pos = camera_info(param)
        pt_trans = np.dot(position - cam_pos, cam_mat.transpose())
        nom_trans = np.dot(normal, cam_mat.transpose())
        train_data = np.hstack((pt_trans, nom_trans))

        train_data_h5 = os.path.join(target_filedir, '{}_sr.h5'.format(strview))

        # img_path = x1_img_path.replace('_x1.png', '.png')
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (224, 224))
        # X, Y, Z = pt_trans.T
        # F = 248
        # h = (-Y)/(-Z)*F + 224/2.0
        # w = X/(-Z)*F + 224/2.0
        # h = np.minimum(np.maximum(h, 0), 223)
        # w = np.minimum(np.maximum(w, 0), 223)
        # img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
        # img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
        # print(img_path.replace('.png', '_prj.png'))
        # cv2.imwrite(img_path.replace('.png', '_prj.png'), img)

        with h5py.File(train_data_h5, 'w') as fp:
            fp.create_dataset('img_ori', data=xori)
            fp.create_dataset('img_x1', data=x1)
            fp.create_dataset('img_x2', data=x2)
            fp.create_dataset('img_x4', data=x4)
            fp.create_dataset('img_x8', data=x8)
            fp.create_dataset('ptnm', data=train_data)

        # break


pool = mp.Pool(10)
pool.map(gen, files)
pool.close()
pool.join()
