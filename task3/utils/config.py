import yaml

from torch.utils.data import DataLoader, Subset

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
    assert mode in ['train', 'val', 'test']

    incl_samples = None
    excl_samples = None

    if mode in ['val', 'train']: # TODO: train/val splitting if necessary
        incl_samples = cfg['data'].get('include_samples', None)
        if incl_samples != None:
            incl_samples = incl_samples.split(',')
        excl_samples = cfg['data'].get('exclude_samples', None)
        if excl_samples != None:
            excl_samples = excl_samples.split(',')
        batch_size = cfg['training'].get('batch_size', 8)

    else:
        batch_size = 1

    dataset = Dataset(
        dataset_folder=cfg['data']['path'],
        include_samples=incl_samples,
        exclude_samples=excl_samples,
        mode=mode,
        img_size=(cfg['data']['resy'], cfg['data']['resx']),
        asp_ratio=(cfg['data']['asp_y'], cfg['data']['asp_x']),
        only_annotated=cfg['data']['only_annotated'],
    )


    # Set up data loader
    shuffle = mode in ['train', 'val'] # in testing shuffling is never used # TODO: set to previous value for submission (only mode == 'train')
    num_workers = cfg['training'].get('num_workers', 0)

    if get_subset:
        # get subset of data for quick training
        subset_idx = range(0, len(dataset), cfg['training'].get('take_every', 20))
        subset = Subset(dataset, subset_idx)

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
