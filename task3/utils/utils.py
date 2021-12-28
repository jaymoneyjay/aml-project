import os


def dir_exists(path):
    return os.path.exists(path)


def file_exists(path):
    return os.path.isfile(path)


def cond_mkdir(path):
    if not dir_exists(path):
        os.makedirs(path)