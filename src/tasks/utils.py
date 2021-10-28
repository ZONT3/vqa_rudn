import os
from argparse import Namespace

import torch


def new_dir_name(prefix: str):
    i = 0
    while os.path.isfile(f'{prefix}{i}/LAST.pth') or os.path.isfile(f'{prefix}{i}/test_predict.json'):
        i += 1
    return f'{prefix}{i}'


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def handle_args(args: Namespace):
    args.output = new_dir_name(args.output)
    args.optimizer = get_optimizer(args.optim)
