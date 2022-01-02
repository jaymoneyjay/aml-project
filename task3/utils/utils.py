import os
from torch import is_tensor
from torchvision import transforms
import numpy as np

from loguru import logger

def dir_exists(path):
    return os.path.exists(path)


def file_exists(path):
    return os.path.isfile(path)


def cond_mkdir(path):
    if not dir_exists(path):
        os.makedirs(path)

def get_ith_element_from_dict_of_tensors(i, dictionary=None):
    """ only shallow copies! """
    if dictionary is None:
        dictionary = {}
    assert type(dictionary) is dict

    copy = {}

    for key in dictionary.keys():
        val = dictionary[key]
        if is_tensor(val):
            copy[key] = val[i]
        elif type(val) is list:
            copy[key] = []
            for tensor in val:
                copy[key].append(tensor[i].item()) # change this for deeper data structures!
            copy[key] = tuple(copy[key])
        else:
            copy[key] = val

    logger.debug(copy)
    return copy

def upscale(frame, img_dims, roi_coord, roi_dims):
    img_height, img_width = img_dims
    roi_height, roi_width = roi_dims
    x, y = roi_coord
    
    top = y
    bottom = img_height - y - roi_height
    
    left = x
    right = img_width - x - roi_width
    
    f_pad = np.pad(frame, ((top, bottom), (left, right)), constant_values=0)
    return f_pad

def get_img_dims(df_meta, name):
    sample_meta = df_meta[df_meta.name == name]
    img_dims = (sample_meta.frame_height.values[0], sample_meta.frame_width.values[0])
    return img_dims