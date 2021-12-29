import os
from torch import is_tensor

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
