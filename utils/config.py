import os
import pprint
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

from utils.logger import create_logger


# dataset root
DATASET_ROOT = "datasets/data"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")
IMAGENET_ROOT = os.path.join(DATASET_ROOT, "imagenet")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "shapenet/ellipsoid/pixel2mesh_aux_4stages.dat")

# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": os.path.join(SHAPENET_ROOT, "pretrained/vgg16-397923af.pth"),
    "resnet50": os.path.join(SHAPENET_ROOT, "pretrained/resnet50-19c8e357.pth"),
    "vgg16p2m": os.path.join(SHAPENET_ROOT, "pretrained/vgg16-p2m.pth"),
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


options = edict()

options.name = 'default'
options.version = None
options.num_workers = 0
options.num_gpus = 1
options.pin_memory = True
options.seed = 1234

options.experiments_dir = "experiments"
options.log_dir = "logs"
options.log_level = "info"
options.summary_dir = "summary"
options.predict_dir = "predict"
options.checkpoint_dir = "checkpoints"
options.checkpoint = None

options.dataset = edict()
options.dataset.name = "shapenet"
# options.dataset.img_dir = "SuperResoRenderingh5_down8_v1"
options.dataset.img_dir = "ShapeNetRenderingh5_v1"
options.dataset.sdf_dir = "SDF_v1"
options.dataset.filelist_train = ["02691156_train.lst"]
options.dataset.filelist_test = ["02691156_test.lst"]
options.dataset.normalization = True
options.dataset.camera_f = [248., 248.]
options.dataset.camera_c = [111.5, 111.5]
options.dataset.mesh_pos = [0., 0., -0.8]
options.dataset.normalization = True
options.dataset.num_classes = 13

# options.dataset.subset_train = "train_small"
# options.dataset.subset_eval = "test_small"

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 4096
options.dataset.shapenet.resize_with_constant_border = False
options.dataset.shapenet.img_alpha = False
options.dataset.shapenet.augcolorfore = False
options.dataset.shapenet.augcolorback = False
options.dataset.shapenet.rot = False

options.dataset.predict = edict()
options.dataset.predict.folder = "predict"

options.model = edict()
options.model.name = "pixel2mesh"
options.model.backbone_pretrained = True

options.model.hidden_dim = 192
options.model.last_hidden_dim = 192
options.model.coord_dim = 3
options.model.backbone = "res18"
options.model.gconv_activation = True
# provide a boundary for z, so that z will never be equal to 0, on denominator
# if z is greater than 0, it will never be less than z;
# if z is less than 0, it will never be greater than z.
options.model.z_threshold = 0
# align with original tensorflow model
# please follow experiments/tensorflow.yml
options.model.align_with_tensorflow = False

options.model.img_size = 224
options.model.map_size = 137
options.model.tanh = False


options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.6e-4
options.loss.weights.edge = 0.3
options.loss.weights.laplace = 0.5
options.loss.weights.move = 0.1
options.loss.weights.constant = 1.
options.loss.weights.chamfer = [1., 1., 1.]
options.loss.weights.chamfer_opposite = 1.
options.loss.weights.reconst = 0.
options.loss.sdf = edict()
options.loss.sdf.coefficient = 1000.
options.loss.sdf.threshold = 0.01
options.loss.sdf.weights = edict()
options.loss.sdf.weights.near_surface = 4.
options.loss.sdf.weights.scale = 10.
# options.loss.weights.normal = 1.6e-4
# options.loss.weights.edge = 0.3
# options.loss.weights.laplace = 0.5
# options.loss.weights.move = 0.1
# options.loss.weights.constant = 1.
# options.loss.weights.chamfer = [1., 1., 1.]
# options.loss.weights.chamfer_opposite = 1.
# options.loss.weights.reconst = 0.

options.train = edict()
options.train.num_epochs = 50
options.train.batch_size = 4
options.train.summary_steps = 50
options.train.checkpoint_steps = 10000
options.train.test_epochs = 1
options.train.use_augmentation = True
options.train.shuffle = True
options.train.backbone_pretrained_model = None  # 'misc/vgg16_nn_encoder.pth'

options.test = edict()
options.test.dataset = []
options.test.summary_steps = 50
options.test.batch_size = 4
options.test.shuffle = False
options.test.weighted_mean = False

options.optim = edict()
options.optim.name = "adam"
options.optim.adam_beta1 = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 1.0e-4
options.optim.wd = 1.0e-5
options.optim.lr_scheduler = "multistep"
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.3
options.optim.lr_gamma = 1.0


def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_options(options_file):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(options_file) as f:
        options_dict = yaml.safe_load(f)
        # do a dfs on `BASED_ON` options files
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                _update_options(os.path.join(os.path.dirname(options_file), base_options))
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def update_options(options_file):
    _update_options(options_file)


def gen_options(options_file):
    def to_dict(ed):
        ret = dict(ed)
        for k, v in ret.items():
            if isinstance(v, edict):
                ret[k] = to_dict(v)
            elif isinstance(v, np.ndarray):
                ret[k] = v.tolist()
        return ret

    cfg = to_dict(options)

    with open(options_file, 'w') as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)


def slugify(filename):
    filename = os.path.relpath(filename, ".")
    if filename.startswith("configs/"):
        filename = filename[len("configs/"):]
    return os.path.splitext(filename)[0].lower().replace("/", "_").replace(".", "_")


def reset_options(options, args, phase='train'):
    if hasattr(args, "batch_size") and args.batch_size:
        options.train.batch_size = options.test.batch_size = args.batch_size
    if hasattr(args, "version") and args.version:
        options.version = args.version
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint
    if hasattr(args, "folder") and args.folder:
        options.dataset.predict.folder = args.folder
    if hasattr(args, "gpus") and args.gpus:
        options.num_gpus = args.gpus
    if hasattr(args, "shuffle") and args.shuffle:
        options.train.shuffle = options.test.shuffle = True
    if hasattr(args, "name") and args.name:
        options.name = args.name
    cwd = os.getcwd()

    if options.version is None:
        prefix = ""
        if args.options:
            prefix = slugify(args.options) + "_"
        options.version = prefix + datetime.now().strftime('%m%d%H%M%S')  # ignore %Y
    options.log_dir = os.path.join(cwd, options.experiments_dir, options.log_dir, options.name)
    print('=> creating {}'.format(options.log_dir))
    os.makedirs(options.log_dir, exist_ok=True)

    options.checkpoint_dir = os.path.join(cwd, options.experiments_dir, options.checkpoint_dir, options.name, options.version)
    if phase != 'predict':
        print('=> creating {}'.format(options.checkpoint_dir))
        os.makedirs(options.checkpoint_dir, exist_ok=True)

    options.summary_dir = os.path.join(cwd, options.experiments_dir, options.summary_dir, options.name, options.version)
    if phase != 'predict':
        print('=> creating {}'.format(options.summary_dir))
        os.makedirs(options.summary_dir, exist_ok=True)

    if phase == 'predict':
        print('=> do not create summary writer for predict')
        writer = None
    else:
        print('=> creating summary writer')
        writer = SummaryWriter(options.summary_dir)

    options.predict_dir = os.path.join(cwd, options.experiments_dir, options.predict_dir, options.name, options.version)
    if phase == 'predict':
        print('=> creating {}'.format(options.predict_dir))
        os.makedirs(options.predict_dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    options_text = pprint.pformat(vars(options))
    logger.info(options_text)

    return logger, writer


if __name__ == "__main__":
    parser = ArgumentParser("Read options and freeze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    update_options(args.input)
    gen_options(args.output)
