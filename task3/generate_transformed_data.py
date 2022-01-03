import task3.utils.config
import task3.utils.utils
import task3.utils.img_utils
from loguru import logger
import importlib
import sys
from tqdm import tqdm

from task3.utils.config import get_data_loader, init

importlib.reload(sys.modules['task3.utils.config'])
importlib.reload(sys.modules['task3.utils.utils'])
from task3.utils.config_full import get_data_loader, init
from task3.utils.data_utils import save_zipped_pickle

cfg = init(config='configs/default.yaml')

# you'd probably call this in train.py
training_loader, validation_loader, test_loader = get_data_loader(cfg, mode='train', get_subset=False)

train_items = [item for item in tqdm(training_loader)]
val_items = [item for item in tqdm(validation_loader)]
test_items = [item for item in tqdm(test_loader)] if test_loader is not None else None
submission_loader = get_data_loader(cfg, mode='submission', get_subset=False)
submission_items = [item for item in tqdm(submission_loader)]

sets = {
    'train': train_items,
    'val': val_items,
    'test': test_items,
    'submission': submission_items,
}

save_zipped_pickle(sets, 'data/transformed_data.pkl')

