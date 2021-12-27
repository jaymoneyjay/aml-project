from task3.utils.data_utils import load_zipped_pickle
from task3.utils.img_utils import get_segment_crop, mask_to_ratio, resize_img

from torchvision import transforms
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from loguru import logger


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
        self.data = self._prepare_files()

        logger.debug('Exclude samples: {}, include samples: {}, applied transforms: {}', self.exclude_samples,
                     self.include_samples, self.transformations)

    def _prepare_files(self):
        data = []

        # unzip and load dataset
        samples = load_zipped_pickle("{}/expert_train_padded.pkl".format(self.dataset_folder)) if not self.is_submission else load_zipped_pickle("{}/expert_test_padded.pkl".format(self.dataset_folder))
        print(samples[0].keys())
        if not self.is_submission:
            # Only use selected dataset
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

        sample_names = []
        for sample in samples:
            sample_names.append(sample['name'])
        logger.debug('Loaded samples: {}', sample_names)

        self.samples = samples

        # flatten samples to obtain dataset
        # TODO KeyError: 'frames' when loading test dataset
        for sample in samples:
            for i in range(sample['video'].shape[-1]):
                frame = sample['video'][:, :, i]
                label = sample['label'][:, :, i] if not self.is_submission and i in sample['frames'] else None
                if not self.only_annotated or label is not None or self.is_submission:
                    data.append({
                        'id': '{}_{}'.format(sample['name'], i),
                        'name': sample['name'], # keep name as it is needed in evaluation function
                        'frame': frame.astype(np.uint8), 
                        'box': sample['roi'] if not self.is_submission else None, # replace with 'roi'
                        'dataset': sample['dataset'] if not self.is_submission else None,
                        'label': label, # bool
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
        name = item['name']
        frame = item['frame']
        label = item['label']
        mask = item['box'] # TODO: ROI predicition included by new dataset and replacing sample key in _init_

        resized_label = None
        resized_frame = None

        if not self.is_submission:
            # crop frame to bounding box, then rescale to target resolution
            new_mask = mask_to_ratio(mask, height=self.asp_ratio[0], width=self.asp_ratio[1])
            cropped_frame = get_segment_crop(frame, mask=new_mask)
            #cropped_frame = get_segment_crop(frame, mask=mask)
            resized_frame = resize_img(cropped_frame, width=self.img_size[1], height=self.img_size[0])

            # crop label to bounding box, then rescale to target resolution
            if label is not None:
                cropped_label = get_segment_crop(label, mask=new_mask)
                #cropped_label = get_segment_crop(label, mask=mask)
                resized_label = resize_img(cropped_label, width=self.img_size[1], height=self.img_size[0])
                resized_label = resized_label.astype(bool) # np.bool depreciated
            # check to see if we are applying any transformations
            if self.transformations is not None:
                # apply the transformations to both image and its mask
                resized_frame = self.transformations(resized_frame)
                resized_label = self.transformations(resized_label)

            assert resized_label.dtype == torch.bool
        else:
            # careful if we resize test set for prediction, we need to scale it back afterwards
            # we should apply toTensor transformation shouldn't we? Otherwise we'll get the
            # RuntimeError: expected scalar type Byte but found Float when trying to run predictions
            # Just make sure that we don't define data augmentation transformation for submission set
            resized_frame = resize_img(frame, width=self.img_size[1], height=self.img_size[0])
            resized_frame = self.transformations(resized_frame)


        if self.is_submission:
            return {
                'id': item['id'],
                'name' : name,
                'frame_cropped': resized_frame,
            }
        else:

            item_out = {
                'id': item['id'],
                'name' : name,
                'frame_cropped': resized_frame,
                'dataset': item['dataset'],
            }

            if resized_label is not None:
                item_out['label_cropped'] = resized_label

            return item_out
