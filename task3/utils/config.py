import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
# changed to import dataset-roi file
from task3.utils.datasetroi import Dataset
from torch import optim
import segmentation_models_pytorch as smp
from torchmetrics import IoU
from torch.nn import BCEWithLogitsLoss, BCELoss
from loguru import logger


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
        
    dataset = Dataset(
        data_cfg=data_cfg,
        mode=mode,
    )

    batch_size = 1
    if mode in ['train']:
        batch_size = data_cfg.get('batch_size', 8)

    # Set up data loader
    shuffle = (mode == 'train') # in testing shuffling is never used # TODO: set to previous value for submission (only mode == 'train')
    num_workers = cfg['training'].get('num_workers', 0)

    subset = None
    if get_subset:
        # get subset of data for quick training
        subset_idx = range(0, len(dataset), cfg['training'].get('take_every', 20))
        subset = Subset(dataset, subset_idx)

    # split train and validation set according to https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    if mode != 'submission':
        #test_split = data_cfg.get('test_split', 0.2)
        validation_split = data_cfg.get('validation_split', 0.2)

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        #test_size = int(np.floor(test_split * dataset_size))
        #train_size = int(np.floor((1 - validation_split) * (dataset_size - test_size)))
        if shuffle:
            np.random.shuffle(indices)
#        train_indices, test_indices, val_indices = indices[:train_size], indices[train_size:(train_size + test_size)], #indices[(train_size + test_size):]
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        logger.debug('Dataset creation: train')
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(subset if get_subset else dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=train_sampler,
                                  )

        logger.debug('Dataset creation: validation')
        valid_sampler = SubsetRandomSampler(val_indices)
        validation_loader = DataLoader(subset if get_subset else dataset, 
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       sampler=valid_sampler
                                      )

        #logger.debug('Dataset creation: test')
        #test_sampler = SubsetRandomSampler(test_indices)
        #test_loader = DataLoader(subset if get_subset else dataset,
        #                               batch_size=batch_size,
        #                               num_workers=num_workers,
        #                               sampler=test_sampler
        #                               )
        
        return train_loader, validation_loader#, test_loader
    
    else: # no sampling needed for test set
        data_loader = DataLoader(
               subset if get_subset else dataset,
               batch_size=batch_size,
               num_workers=num_workers,
               shuffle=shuffle,
           )
            
        return data_loader


def get_model(cfg):
    model = smp.Unet(**cfg['model'].get('smp-unet'))
    params = cfg['model'].get('smp-unet')
    logger.info(f'model params set to: {params}')

    return model.to(device=cfg['device'])

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
        assert cfg['model']['smp-unet']['activation'] == None, f'Last layer of the Unet model should not be sigmoid \
        if you are using {loss}.'

        criterion = BCEWithLogitsLoss(pos_weight=None)

    elif loss == 'bce':    
        assert cfg['model']['smp-unet']['activation'] == 'sigmoid', f'Last layer of the Unet model should be sigmoid \
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
    pass

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
