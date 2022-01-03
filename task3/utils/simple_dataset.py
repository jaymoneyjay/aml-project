from task3.utils.data_utils import load_zipped_pickle
import task3.utils.img_utils

from torchvision import transforms
import torch
from torch.utils.data.dataloader import default_collate

from loguru import logger

import random

from importlib import reload
import sys
reload(sys.modules['task3.utils.img_utils'])
from tqdm import tqdm
from task3.utils.img_utils import *

class Dataset(torch.utils.data.Dataset):
    """ Dataset class"""

    def __init__(self, data_cfg=None, mode='train', img_transforms=None, device='cpu'):
        """ Initialization of the the dataset.

        Args:
            data_cfg (dict or None): config used for initializing this Dataset class
            mode (str): train or test mode, train automatically returns a train/val split dataloader
        """
        if data_cfg is None:
            data_cfg = {}

        assert mode in ['train', 'val', 'test', 'submission']
        self.mode = mode

        # read data config
        self.device = device
        self.dataset_folder = 'data'
        self.dataset_path = "{}/{}".format(self.dataset_folder, 'transformed_data.pkl')
        self.samples = []
        self.data = self._prepare_files()

    def _prepare_files(self):
        samples = load_zipped_pickle(self.dataset_path)

        logger.debug(samples.keys())

        samples = samples[self.mode]
        corrected_samples = []
        for i, sample in enumerate(tqdm(samples)):
            sample['frame_cropped'] = sample['frame_cropped'][0, :, :, :]
            label = sample.get('label_cropped', None)
            if label is not None:
                sample['label_cropped'] = sample['label_cropped'][0, :, :, :]
            corrected_samples.append(sample)
            if self.mode == 'train' and i > 200:
                break
        return corrected_samples

    def __len__(self):
        """ Returns the length of the dataset. """

        return len(self.data)

    def __getitem__(self, idx):
        """ Returns an item of the dataset. But should actually return frames, label to be easily integrated with other libraries.
        Args:
            idx (int): data ID.
        """
        return self.data[idx]
