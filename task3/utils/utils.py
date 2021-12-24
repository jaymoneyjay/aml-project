import numpy as np
import torch
from random import seed
from .logger import logger_init
from .config import load_config
import os

def init(config='configs/default.yaml'):
    # fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    seed(42)

    cfg = load_config(config)

    # set logger format
    logger_init(cfg['application'].get('log_level', 'DEBUG'))

    return cfg

def cond_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)