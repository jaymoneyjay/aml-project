import numpy as np
import torch
from random import seed
from .logger import logger_init
from .config import load_config
import os

def init(config='configs/default.yaml', checkpoint=None):
    # fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    seed(42)

    # set logger format
    logger_init()

    return load_config(config)

def cond_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)