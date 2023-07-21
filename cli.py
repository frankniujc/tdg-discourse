import argparse
import logging
import random
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os

from temporal_ie import Trainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, not option.startswith('--d'))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-type',
        choices=['tdg', 'tdg-graph'], default='tdg')

    parser.add_argument('--data-path', type=Path, default=Path('data'))
    parser.add_argument('--cache-dir', type=Path, default='/tmp/')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--disable-tqdm', action='store_true')
    parser.add_argument('--monitor-metric', default='f1')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('-m', '--model-name-or-path', default='roberta-base')
    parser.add_argument('--optimizer',
        choices=['Adam', 'AdamW', 'RMSprop', 'SGD'], type=str, default='AdamW')
    parser.add_argument('-ma', '--model-arch', default='baseline')

    # optimizer arguments
    hyperparam = parser.add_argument_group('hyperparam')
    hyperparam.add_argument('--batch-size', type=int, default=16)
    hyperparam.add_argument('--learning-rate', type=float, default=1e-5)

    # GPU arguments
    parser.add_argument(
        '--mg', '--use-multi-gpus',
        '--dmg', '--disable-multi-gpus',
        dest='multi_gpus', action=NegateAction, nargs=0, default=False)
    ddp_args = parser.add_argument_group('ddp')
    ddp_args.add_argument('--rank', default=-1)

    # TDG arguments
    tdg_args = parser.add_argument_group('tdg')
    tdg_args.add_argument('--relation-type', choices=['t2t', 'e2t', 'e2e', 'joint'], default='joint')
    tdg_args.add_argument('--unlabelled', action='store_true')

    args = parser.parse_args()
    logging.info(f'args: {args}')
    return args

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    logging.info(f'init process group for {rank}')
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logging.info('done')

def main():
    args = parse_args()
    if args.multi_gpus:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))
        world_size = torch.cuda.device_count()
        mp.spawn(dist_main, args=(world_size, args), nprocs=world_size)
    else:
        run_trainer(args)

def dist_main(rank, world_size, args):
    ddp_setup(rank, world_size)
    args.rank = rank
    args.num_gpus = world_size
    run_trainer(args)

def run_trainer(args):
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

if __name__ == '__main__':
    main()