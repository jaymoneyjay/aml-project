import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from .dataset import Dataset
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

    incl_samples = None
    excl_samples = None
    transformations = None
    

    if mode in ['train']: # TODO: train/val splitting if necessary
        incl_samples = cfg['data'].get('include_samples', None)
        if incl_samples != None:
            incl_samples = incl_samples.split(',')
        excl_samples = cfg['data'].get('exclude_samples', None)
        if excl_samples != None:
            excl_samples = excl_samples.split(',')
        batch_size = cfg['training'].get('batch_size', 8)

    else:
        batch_size = 1
    
    # TODO add more transformations, i.e. augmentation
    if cfg['data'].get('transforms', True):
        transformations = transforms.Compose([transforms.ToTensor()]) # transform to Tensor and 0-255 -> 0-1
        
    dataset = Dataset(
        dataset_folder=cfg['data']['path'],
        include_samples=incl_samples,
        exclude_samples=excl_samples,
        mode=mode,
        img_size=(cfg['data']['resy'], cfg['data']['resx']),
        asp_ratio=(cfg['data']['asp_y'], cfg['data']['asp_x']),
        only_annotated=cfg['data']['only_annotated'],
        transformations=transformations,
    )

    # Set up data loader
    shuffle = (mode == 'train') # in testing shuffling is never used # TODO: set to previous value for submission (only mode == 'train')
    num_workers = cfg['training'].get('num_workers', 0)

    if get_subset:
        # get subset of data for quick training
        subset_idx = range(0, len(dataset), cfg['training'].get('take_every', 20))
        subset = Subset(dataset, subset_idx)

    # split train and validation set according to https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    if mode != 'test':
        validation_split = cfg['data'].get('validation_split', 0.2)
        shuffle_dataset = cfg['data'].get('shuffle_dataset', True)
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle :
            np.random.seed(random_seed)
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
