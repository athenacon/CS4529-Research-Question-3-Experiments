import torch

from torchvision import datasets, transforms

from trainer import Trainer
from config import get_config
from utils import prepare_dirs
from data_loader import get_test_loader, get_train_valid_loader, VIEWPOINT_EXPS


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ref https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/7
seed_value = 42
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}

    # instantiate data loaders
    if config.is_train:
        print("Training")
        data_loader = get_train_valid_loader(
            config.data_dir, config.dataset, config.batch_size,
            config.random_seed, config.exp, config.valid_size,
            config.shuffle, **kwargs
        )
        print("Data loader: ")
    else:
        data_loader = get_test_loader(
            config.data_dir, config.dataset, config.batch_size, config.exp, config.familiar,
            **kwargs
        )
        print(config.exp, config.familiar)
    # instantiate trainer
    trainer = Trainer(config, data_loader)
    
    if config.is_train:
        print("Training started")
        trainer.train()
    else:
        trainer.test()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
