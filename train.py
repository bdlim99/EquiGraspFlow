import argparse
from datetime import datetime
from omegaconf import OmegaConf
import os
from tensorboardX import SummaryWriter
import logging
import yaml
import random
import numpy as np
import torch

from loaders import get_dataloader
from models import get_model
from losses import get_losses
from utils.optimizers import get_optimizer
from metrics import get_metrics
from utils.logger import Logger
from trainers import get_trainer


def main(cfg, writer):
    # Setup seed
    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Setup dataloader
    dataloaders = {}

    for split in ['train', 'val']:
        dataloaders[split] = get_dataloader(split, cfg.data[split])

    # Setup model
    model = get_model(cfg.model).to(cfg.device)

    # Setup losses
    losses = get_losses(cfg.losses)

    # Setup optimizer
    optimizer = get_optimizer(cfg.optimizer, model.parameters())

    # Setup metrics
    metrics = get_metrics(cfg.metrics)

    # Setup logger
    logger = Logger(writer)

    # Setup trainer
    trainer = get_trainer(cfg.trainer, cfg.device, dataloaders, model, losses, optimizer, metrics, logger)

    # Start learning
    trainer.run()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='train_results')
    parser.add_argument('--run', type=str, default=datetime.now().strftime('%Y%m%d-%H%M'))

    args = parser.parse_args()

    # Load and print config
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    # Setup device
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    # Setup logdir
    config_filename = os.path.basename(args.config)
    config_basename = os.path.splitext(config_filename)[0]

    logdir = os.path.join(args.logdir, config_basename, args.run)

    # Setup tensorboard writer
    writer = SummaryWriter(logdir)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(logdir, 'logging.log'),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG
    )

    # Print logdir
    print(f"Result directory: {logdir}")
    logging.info(f"Result directory: {logdir}")

    # Save config
    config_path = os.path.join(logdir, config_filename)
    yaml.dump(yaml.safe_load(OmegaConf.to_yaml(cfg)), open(config_path, 'w'))

    print(f"Config saved as {config_path}")
    logging.info(f"Config saved as {config_path}")

    main(cfg, writer)
