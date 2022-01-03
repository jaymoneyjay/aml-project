from importlib import reload
import sys
import yaml
import numpy as np
import torch
import random
from functools import partial
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
# changed to import dataset-roi file
from task3.utils.dataset import Dataset
import task3.utils.transforms_utils
from torch import optim
from torchvision import transforms
import segmentation_models_pytorch as smp
from torchmetrics import IoU
from torch.nn import BCEWithLogitsLoss, BCELoss
from loguru import logger
from task3.utils.logger import logger_init

import task3.utils.dataset
reload(sys.modules['task3.utils.dataset'])
from task3.utils.simple_dataset import Dataset

def init(config='configs/default.yaml'):
    # fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    cfg = load_config(config)

    # set logger format
    logger_init(cfg['application'].get('log_level', 'DEBUG'))

    return cfg

@logger.catch
def load_config(config):
    """ Loads configuration file.

    Returns:
        cfg (dict): configuration file
    """
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_data_loader(cfg, mode='train', get_subset=False):
    """ mode train automatically returns two data loaders, one for training and one for validation.
    split is define in yaml.
    """
    assert mode in ['train', 'submission']
    data_cfg = cfg['data']

    device = torch.device(cfg.get('device', 'cpu'))
    logger.debug('selected device: {}', device)

    batch_size = 1
    if mode in ['train']:
        batch_size = data_cfg.get('batch_size', 8)

    # Set up data loader
    num_workers = cfg['training'].get('num_workers', 0)

    subset = None

    # split train and validation set according to https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    if mode != 'submission':

        logger.debug('Dataset creation: train')
        train_dataset = Dataset(
            data_cfg=data_cfg,
            mode='train',
            device=device,
        )
        train_loader = DataLoader(subset if get_subset else train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  )

        logger.debug('Dataset creation: validation')
        val_dataset = Dataset(
            data_cfg=data_cfg,
            mode='val',
            device=device,
        )
        validation_loader = DataLoader(subset if get_subset else val_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  )

        test_loader = None
        if data_cfg['test_split'] > 0:
            logger.debug('Dataset creation: test')
            test_dataset = Dataset(
                data_cfg=data_cfg,
                mode='test',
                device=device,
            )
            test_loader = DataLoader(subset if get_subset else test_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      drop_last=True,
                                      shuffle=True,
                                      )
        return train_loader, validation_loader, test_loader
    
    else:  # no sampling needed for test set
        logger.debug('Dataset creation: submission')
        dataset = Dataset(
            data_cfg=data_cfg,
            mode='submission',
            device=device,
        )

        data_loader = DataLoader(
               subset if get_subset else dataset,
               batch_size=batch_size,
               num_workers=num_workers,
               shuffle=False,
           )

        return data_loader


def get_model(cfg):
    params = cfg['model'].get('params')

    if cfg['model']['name'] == 'smp-unet':
        model = smp.Unet(**params)

    elif cfg['model']['name'] == 'smp-unet-plusplus':
        model = smp.UnetPlusPlus(**params)

    elif cfg['model']['name'] == 'smp-unet-deepv3+':
        model = smp.DeepLabV3Plus(**params)

    else:
        raise Exception('Not supported.')    


    logger.info(f'model params set to: {params}')

    return model.to(device=torch.device(cfg['device']))

def get_optimizer(model, cfg):
    """ Create an optimizer. """

    if cfg['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg['training'].get('lr', 1e-4),
                              momentum=cfg['training'].get('momentum', 0.9))
    elif cfg['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=model.parameters(),
                                          lr=cfg['training'].get('lr', 5e-5),
                                          weight_decay=0)
    else:
        raise Exception('Not supported.')

    return optimizer

def get_loss(model, cfg):
    """Checks that last layer fits the choice of loss/ criterion."""
    
    loss = cfg['training']['loss']
    
    # BCEWithLogits = Sigmoid + BCELoss -> apparently numerically more stable 
    if loss == 'bcewithlogitsloss':
        assert cfg['model']['params']['activation'] == None, f'Last layer of the Unet model should not be sigmoid \
        if you are using {loss}.'

        criterion = BCEWithLogitsLoss(pos_weight=None)

    elif loss == 'bce':    
        assert cfg['model']['params']['activation'] == 'sigmoid', f'Last layer of the Unet model should be sigmoid \
        if you are using {loss}.'

        criterion = BCELoss()

    elif loss == 'jaccard':
        criterion = smp.utils.losses.JaccardLoss()
    
    elif loss == 'dice':
        criterion = smp.utils.losses.DiceLoss()
    else:
        raise Exception('Not supported.')
        
    logger.info(f'Using {criterion} as loss function.')

    return criterion

def get_lrscheduler(optimizer, cfg):
    
    if cfg['training']['lr_scheduler'] == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    elif cfg['training']['lr_scheduler'] == 'reduceonplateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                   min_lr=0, eps=1e-08, verbose=False)
    else:
        raise Exception('Not supported.')
    
    return lr_scheduler

def get_trainer(model, vis_dir, cfg, optimizer=None):
    """ Create a trainer instance. """

    if cfg['trainer'] == 'conv':
        # trainer = ...
        pass
    if cfg['trainer'] == 'hmr':
        # trainer = ...
        pass
    else:
        raise Exception('Not supported.')
    return None