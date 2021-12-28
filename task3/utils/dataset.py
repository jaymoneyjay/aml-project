from importlib import reload
import sys

from torchvision import transforms
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from loguru import logger

from task3.utils.utils import file_exists
import task3.utils.data_utils, task3.utils.img_utils, task3.utils.roi_prediction
reload(sys.modules['task3.utils.data_utils'])
reload(sys.modules['task3.utils.img_utils'])
reload(sys.modules['task3.utils.roi_prediction'])

from task3.utils.data_utils import load_zipped_pickle, save_zipped_pickle
from task3.utils.img_utils import get_segment_crop, mask_to_ratio, resize_img, normalize_expert_dimensions
from task3.utils.roi_prediction import get_box_for_video, get_windows_and_heatmap


class Dataset(torch.utils.data.Dataset):
    """ Dataset class"""

    def __init__(self, data_cfg=None, mode='train'):
        """ Initialization of the the dataset.

        Args:
            data_cfg (dict or None): config used for initializing this Dataset class
            mode (str): train or test mode, train automatically returns a train/val split dataloader
        """
        if data_cfg is None:
            data_cfg = {}

        transformations = None
        if data_cfg.get('transforms', True): # TODO add more transformations, i.e. augmentation
            transformations = transforms.Compose([transforms.ToTensor()])  # transform to Tensor and 0-255 -> 0-1

        incl_samples = data_cfg.get('include_samples', None)
        excl_samples = data_cfg.get('exclude_samples', None)
        if mode in ['train']:  # TODO: train/val splitting if necessary
            if incl_samples is not None:
                incl_samples = incl_samples.split(',')
            if excl_samples is not None:
                excl_samples = excl_samples.split(',')

        # read data config
        self.dataset_folder = data_cfg.get('path', 'data')
        self.train_data_path = "{}/train.pkl".format(self.dataset_folder)
        self.submission_data_path = "{}/test.pkl".format(self.dataset_folder)
        self.exclude_samples = excl_samples
        self.include_samples = incl_samples
        self.mode = mode
        self.dataset = data_cfg.get('dataset', None)
        self.is_submission = (mode == 'submission')
        self.img_size = (data_cfg.get('resy', 30), data_cfg.get('resx', 40)) # output of image cropping
        self.asp_ratio = (data_cfg.get('asp_y', 3), data_cfg.get('asp_x', 4))
        self.only_annotated = data_cfg.get('only_annotated', True)
        self.transformations = transformations
        self.samples = []

        # Load heatmap for box predictions
        if self.is_submission:
            box_heatmap_path = "{}/{}".format(self.dataset_folder,
                                              data_cfg.get('box_heatmap_filename', 'box_heatmap.pkl'))
            logger.info('Looking for heatmap "{}"', box_heatmap_path)
            if file_exists(box_heatmap_path):
                w_and_h = load_zipped_pickle(box_heatmap_path)
                self.heatmap = w_and_h['heatmap']
                self.windows = w_and_h['windows']
                logger.info('Loaded heatmap file "{}"', box_heatmap_path)
            else:
                train_samples = load_zipped_pickle(self.train_data_path)
                expert_samples = list(filter(lambda d: d['dataset'] == 'expert', train_samples))
                normalize_expert_dimensions(expert_samples)
                w_and_h = get_windows_and_heatmap(expert_samples)
                self.heatmap = w_and_h['heatmap']
                self.windows = w_and_h['windows']
                save_zipped_pickle(w_and_h, box_heatmap_path)
                logger.info('New heatmap created under "{}"', box_heatmap_path)

        self.data = self._prepare_files()

        logger.debug('Exclude samples: {}, include samples: {}, applied transforms: {}', self.exclude_samples,
                     self.include_samples, self.transformations)

    def _prepare_files(self):
        data = []

        # unzip and load dataset

        samples = load_zipped_pickle(self.train_data_path) if not self.is_submission else load_zipped_pickle(self.submission_data_path)

        # Only use selected samples from dataset
        if not self.is_submission:
            if self.dataset is not None:
                samples = list(filter(lambda d: d['dataset'] == self.dataset, samples))

            # remove unwanted samples --> not efficient but allows for warnings
            if self.exclude_samples:
                for name in self.exclude_samples:
                    try:
                        sample_idx = next((index for (index, d) in enumerate(samples) if d["name"] == name), None)
                        del samples[sample_idx]
                    except ValueError:
                        logger.warning('Sample name "{}" not found in configured dataset.', name)

            # only pick wanted samples --> not efficient but allows for warnings
            if self.include_samples:
                samples_temp = []
                for name in self.include_samples:
                    try:
                        sample_idx = next((index for (index, d) in enumerate(samples) if d["name"] == name), None)
                        samples_temp.append(samples[sample_idx])
                    except:
                        logger.warning('Sample name "{}" not found in configured dataset.', name)
                samples = samples_temp

        logger.debug('Loaded samples: {}', [sample['name'] for sample in samples])

        self.samples = samples

        # flatten each sample (video -> frame) to obtain dataset
        for sample in samples:
            video = sample['video']
            box = get_box_for_video(video, self.windows, self.heatmap, w_dims=self.img_size) if self.is_submission else sample['box']
            logger.debug('box: {}', box)

            for i in range(video.shape[-1]):
                frame = video[:, :, i]
                label = sample['label'][:, :, i] if not self.is_submission and i in sample['frames'] else None
                if not self.only_annotated or label is not None or self.is_submission:
                    data.append({
                        'id': '{}_{}'.format(sample['name'], i),
                        'frame': frame.astype(np.uint8), 
                        'box': box,
                        'dataset': sample['dataset'] if not self.is_submission else None,
                        'label': label,  # bool
                    })

        return data

    def __len__(self):
        """ Returns the length of the dataset. """

        return len(self.data)

    def __getitem__(self, idx):
        """ Returns an item of the dataset. But should actually return frames, label to be easily integrated with other libraries.
        Args:
            idx (int): data ID.
        """

        item = self.data[idx]
        frame = item['frame']
        label = item['label']
        mask = item['box'] # TODO: ROI predicition

        resized_label = None
        resized_frame = None

        if not self.is_submission:
            # crop frame to bounding box, then rescale to target resolution
            new_mask = mask_to_ratio(mask, height=self.asp_ratio[0], width=self.asp_ratio[1])
            cropped_frame = get_segment_crop(frame, mask=new_mask)
            resized_frame = resize_img(cropped_frame, width=self.img_size[1], height=self.img_size[0])

            # crop label to bounding box, then rescale to target resolution
            if label is not None:
                cropped_label = get_segment_crop(label, mask=new_mask)
                resized_label = resize_img(cropped_label, width=self.img_size[1], height=self.img_size[0])
                resized_label = resized_label.astype(bool) # np.bool depreciated

            # check to see if we are applying any transformations
            if self.transformations is not None:
                # apply the transformations to both image and its mask
                resized_frame = self.transformations(resized_frame)
                resized_label = self.transformations(resized_label)

            assert resized_label.dtype == torch.bool
        else:
            resized_frame = resize_img(frame, width=self.img_size[1], height=self.img_size[0])

        # new_mask_props = get_box_props(new_mask)

        if self.is_submission:
            return {
                'id': item['id'],
                'frame_cropped': resized_frame,
            }
        else:
            item_out = {
                'id': item['id'],
                'frame_cropped': resized_frame,
                'dataset': item['dataset'],
            }

            if resized_label is not None:
                item_out['label_cropped'] = resized_label

            return item_out
