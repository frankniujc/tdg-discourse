import json
import random
import logging
from pathlib import Path
from argparse import Namespace

from tqdm import tqdm

import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .data import setup_data
from .models import setup_model
from .metrics import setup_metrics

class Trainer:
    def __init__(self, args):
        self.args = args

    @classmethod
    def load_from_checkpoint(cls, path, update_kwargs={}):
        with open(path / 'args.json') as open_file:
            args = json.load(open_file)
        args['from_checkpoint'] = path
        args['data_path'] = Path(args['data_path'])
        args['checkpoint_dir'] = Path(args['checkpoint_dir'])
        args['cache_dir'] = Path(args['cache_dir'])

        for name, val in update_kwargs.items():
            args[name] = val

        args = Namespace(**args)

        trainer = cls(args)
        trainer.setup()
        trainer._load_from_checkpoint()
        return trainer

    def _load_from_checkpoint(self):
        self.model.load_state_dict(torch.load(self.args.from_checkpoint / 'checkpoint.pt'), strict=False)

    def setup(self, reusable=None):
        self.set_seed()
        self.set_device()

        self.setup_reusable(reusable)

        if self.args.unlabelled:
            self.output_size = 2

        model, self.train_epoch, self.setup_training = setup_model(self.args, self.output_size)
        self.model = model.to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()

        if self.args.multi_gpus:
            self.setup_ddp()

    def setup_reusable(self, reusable=None):
        if reusable:
            self.tokenizer, self.output_size, self.dataloaders, self.metrics = reusable
        else:
            self.set_tokenizer()
            self.output_size, self.dataloaders = setup_data(self.args, self.tokenizer)
            self.metrics = setup_metrics(self.args)
            return self.tokenizer, self.output_size, self.dataloaders, self.metrics

    def setup_ddp(self):
        self.model = DDP(self.model,
            device_ids=[self.args.rank],
            output_device=self.args.rank,
            find_unused_parameters=True,
        )

        for name, dataloader in self.dataloaders.__dict__.items():
            dataset = dataloader.dataset
            ddp_dataloader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                pin_memory=True,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            self.dataloaders.__dict__[name] = ddp_dataloader

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        logging.info(f"Set random seed to {self.args.seed}")

    def set_device(self):
        if self.args.multi_gpus:
            self.device = torch.device(self.args.rank)
        else:
            device = torch.device(self.args.device)
            self.device = device

    def set_tokenizer(self):
        do_lower_case = 'uncased' in self.args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base',
            do_lower_case=do_lower_case, use_fast=True)

    def process_batch(self, batch):
        processed_batch = {}
        for name, item in batch.items():
            if isinstance(item, torch.Tensor):
                processed_batch[name] = item.to(self.device)
            else:
                processed_batch[name] = item
        return processed_batch

    def train(self):
        self.setup_training(self)

        self.results = {'dev': {}}
        self.best_epoch = -100

        for epoch in range(self.args.epochs):
            if not self.args.multi_gpus or self.args.rank == 0:
                logging.info(f'EPOCH {epoch}')
            self.train_epoch(self, epoch)
            self.evaluate_and_maybe_save(epoch)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_labels, all_outputs, all_label_ids = [], [], []

        for batch in tqdm(dataloader, disable=self.args.disable_tqdm):
            batch = self.process_batch(batch)
            labels = batch.pop('labels')
            label_ids = batch.pop('label_ids')
            outputs = self.model(**batch)

            all_labels += labels
            all_outputs += outputs
            all_label_ids += label_ids

        all_labels = torch.stack(all_labels)
        all_outputs = torch.stack(all_outputs)
        all_label_ids = torch.stack(all_label_ids)

        if self.args.multi_gpus:
            all_outputs, all_labels, all_label_ids = self.all_gather(
                all_outputs, all_labels, all_label_ids)

        metrics = self.metrics(all_outputs, all_labels, all_label_ids)

        if not self.args.multi_gpus or self.args.rank == 0:
            for name, metric in metrics.items():
                logging.info(f'{name}: {metric}')

        return metrics

    def all_gather(self, all_outputs, all_labels, all_label_ids):
        all_outputs_lst = [torch.zeros_like(all_outputs) for _ in range(self.args.num_gpus)]
        dist.all_gather(all_outputs_lst, all_outputs)
        all_outputs = torch.cat(all_outputs_lst, dim=0)

        all_labels_lst = [torch.zeros_like(all_labels) for _ in range(self.args.num_gpus)]
        dist.all_gather(all_labels_lst, all_labels)
        all_labels = torch.cat(all_labels_lst, dim=0)

        all_label_ids_lst = [torch.zeros_like(all_label_ids) for _ in range(self.args.num_gpus)]
        dist.all_gather(all_label_ids_lst, all_label_ids)
        all_label_ids = torch.cat(all_label_ids_lst, dim=0)

        return all_outputs, all_labels, all_label_ids

    def evaluate_and_maybe_save(self, epoch):
        if self.args.multi_gpus:
            self.dataloaders.dev.sampler.set_epoch(epoch)
        result = self.evaluate(self.dataloaders.dev)

        if not self.args.multi_gpus or self.args.rank == 0:
            self.results['dev'][epoch] = result
            epoch_score = self.results['dev'][epoch][self.args.monitor_metric]

            if epoch_score > self.best_epoch:
                self.best_epoch = epoch_score
                if self.args.checkpoint_dir:
                    self.save_model()

    def save_model(self):
        with open(self.args.checkpoint_dir / 'args.json', 'wt') as open_file:
            json.dump(get_arg_dict(self.args), open_file, indent=2)

        path = self.args.checkpoint_dir / 'checkpoint.pt'
        torch.save(self.model.state_dict(), path)