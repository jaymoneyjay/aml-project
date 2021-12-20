import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from task3.utils.dataset import Dataset
from loguru import logger
from torch import optim

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
    assert mode in ['train', 'test']
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
    if mode != 'test':
        validation_split = data_cfg.get('validation_split', 0.2)

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(subset if get_subset else dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  #shuffle=shuffle, # shuffle is mutually exclusive with sampler option
                                  sampler=train_sampler,
                                  )
        
        valid_sampler = SubsetRandomSampler(val_indices)
        validation_loader = DataLoader(subset if get_subset else dataset, 
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       #shuffle=shuffle, # shuffle is mutually exclusive with sampler option
                                       sampler=valid_sampler
                                      )
        
        return train_loader, validation_loader
    
    else: # no sampling needed for test set
        data_loader = DataLoader(
               subset if get_subset else dataset,
               batch_size=batch_size,
               num_workers=num_workers,
               shuffle=shuffle,
           )
            
        return data_loader


def get_model(cfg):
    # model = create_model(cfg)
    #return model.to(device=cfg['device'])
    pass

def get_optimizer(model, cfg):
    """ Create an optimizer. """

    if cfg['training']['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg['training']['optimizer'].get('lr', 1e-4))
    elif cfg['training']['optimizer']['name'] == 'ADAM':
        optimizer = optim.Adam(params=model.parameters(),
                                          lr=cfg['training']['optimizer'].get('lr', 5e-5),
                                          weight_decay=0)
    else:
        raise Exception('Not supported.')

    return optimizer


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
