import os
from glob import glob
from os.path import join, exists, basename, splitext
from task3.utils.data_utils import load_zipped_pickle
from task3.utils.img_utils import np_to_opencv, get_segment_crop, mask_to_ratio, get_box_props, resize_img, show_img

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from random import choice

from loguru import logger


class Dataset(torch.utils.data.Dataset):
    """ Dataset class"""

    def __init__(self, mode='train', img_size=(112, 112), dataset_folder='data', exclude_samples=[], include_samples=[], only_annotated=True):
        """ Initialization of the the dataset.

        Args:
            dataset_folder (str): dataset folder
            exclude_samples (None/list): list of samples to discard (training only)
            include_samples (None/list): list of samples to be used (training only). include_samples overrides exclude_samples
            mode (str): train, val, or test mode
            img_size (tuple): image resolution. Tuple of integers (height, width).
            only_annotated (Bool): If true, only frames with a label are added to the dataset
        """
        # Attributes
        logger.debug('Exclude samples: {}, include samples: {}', exclude_samples, include_samples)

        self.dataset_folder = dataset_folder
        self.exclude_samples = exclude_samples
        self.include_samples = include_samples
        self.mode = mode
        self.is_test = (mode == 'test')
        self.img_size = img_size # output of image cropping
        self.only_annotated = only_annotated

        self.samples = []
        self.data = self._prepare_files()

    def _prepare_files(self):
        data = []

        # unzip and load dataset
        samples = load_zipped_pickle("{}/train.pkl".format(self.dataset_folder)) if not self.is_test else load_zipped_pickle("{}/test.pkl".format(self.dataset_folder))

        if not self.is_test:
            # remove unwanted samples --> not efficient but allows for warnings
            if self.exclude_samples:
                for name in self.exclude_samples:
                    try:
                        sample_idx = next((index for (index, d) in enumerate(samples) if d["name"] == name), None)
                        del samples[sample_idx]
                    except ValueError:
                        logger.warning('Sample name "{}" invalid.', name)

            # only pick wanted samples --> not efficient but allows for warnings
            if self.include_samples:
                samples_temp = []
                for name in self.include_samples:
                    try:
                        sample_idx = next((index for (index, d) in enumerate(samples) if d["name"] == name), None)
                        samples_temp.append(samples[sample_idx])
                    except:
                        logger.warning('Sample name "{}" invalid.', name)
                samples = samples_temp

        sample_names = []
        for sample in samples:
            sample_names.append(sample['name'])
        logger.debug('Loaded samples: {}', sample_names)

        self.samples = samples

        # flatten samples to obtain dataset
        for sample in samples:
            for i in range(sample['video'].shape[-1]):
                frame = sample['video'][:, :, i]
                label = sample['label'][:, :, i] if i in sample['frames'] else None
                if not self.only_annotated or label is not None:
                    data.append({
                        'id': '{}_{}'.format(sample['name'], i),
                        'frame': frame.astype(np.uint8),
                        'box': sample['box'],
                        'dataset': sample['dataset'],
                        'label': label,
                    })

        return data

    def __len__(self):
        """ Returns the length of the dataset. """

        return len(self.data)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.mode == 'train':
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            #### noise factor = 0.4
            noise_factor = 0.4
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # rot factor = 30
            ####rot = choice([0, 90, 180, 270])
            rot_factor = 30
            rot = min(2 * rot_factor,
                      max(-2 * rot_factor, np.random.randn() * rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            #### scale factor = 0.25
            scale_factor = 0.25
            sc = min(1 + scale_factor,
                     max(1 - scale_factor, np.random.randn() * scale_factor + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        return flip, pn, rot, sc

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): data ID.
        """

        item = self.data[idx]
        frame = item['frame']
        label = item['label']
        mask = item['box']

        # crop frame to bounding box, then rescale to target resolution
        new_mask = mask_to_ratio(mask, height=3, width=4)
        cropped_frame = get_segment_crop(frame, mask=new_mask)
        resized_frame = resize_img(cropped_frame, width=self.img_size[0], height=self.img_size[1])

        # crop label to bounding box, then rescale to target resolution
        resized_label = None
        if label is not None:
            cropped_label = get_segment_crop(label, mask=new_mask)
            resized_label = resize_img(cropped_label, width=self.img_size[0], height=self.img_size[1])

        item_out = {
            'id': item['id'],
            # 'frame': frame,
            'frame_cropped': resized_frame,
            'dataset': item['dataset'],
            # 'label': item['label'],
            # 'label_cropped': resized_label,
        }
        if label is not None:
            item_out['label_cropped'] = resized_label

        return item_out
