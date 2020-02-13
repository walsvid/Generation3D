import logging
import os
import random
import numpy as np
import torch


def create_logger(cfg, phase='train'):
    cwd = os.getcwd()
    log_file = '{}_{}.log'.format(cfg.version, phase)
    final_log_file = os.path.join(cwd, cfg.experiments_dir, cfg.log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    if cfg.log_level == "info":
        logger.setLevel(logging.INFO)
    elif cfg.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError("Log level has to be one of info and debug")
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
