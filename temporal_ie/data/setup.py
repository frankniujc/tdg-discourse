import torch
import logging

from .tdg.pairwise import TDGPairDataset

datasets = {
    'tdg': TDGPairDataset,
}

def setup_data(args, tokenizer):
    dataset_type = datasets[args.dataset_type]
    dataset, dataloaders = dataset_type.setup_dataset(args, tokenizer)
    output_size = dataset.output_size
    return output_size, dataloaders