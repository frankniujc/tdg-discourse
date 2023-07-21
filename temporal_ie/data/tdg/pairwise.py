import itertools
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..utils import DictDataset
from .utils import CONVERT_TOKENS, T2T_LABELS, E2T_LABELS, E2E_LABELS
from .utils import detokenize, get_spans

def get_pairs(timexs, events, edge_map, sent_para_map, relation_type):
    if relation_type == 't2t':
        for pair in itertools.permutations(timexs, 2):
            yield pair
    elif relation_type == 'e2t':
        for pair in itertools.product(events, timexs):
            (s1, b1, e1), (s2, b2, e2) = pair
            p1 = sent_para_map[s1]
            p2 = sent_para_map[s2]
            if p1 == p2:
                yield pair
            elif abs(p1 - p2) == 1:
                yield pair
            elif p2 in [0,1,-1]:
                yield pair
            else:
                if pair in edge_map:
                    raise ValueError(f'Pair {pair} should not be here!')

    elif relation_type == 'e2e':
        for pair in itertools.permutations(events, 2):
            # yield pair
            (s1, b1, e1), (s2, b2, e2) = pair
            p1 = sent_para_map[s1]
            p2 = sent_para_map[s2]
            if p1 == p2:
                if pair in edge_map:
                    for i in range(10):
                        yield pair
                else:
                    yield pair
            elif abs(p1 - p2) == 1:
                if pair in edge_map:
                    for i in range(10):
                        yield pair
                else:
                    yield pair
            elif p2 in [0,1,-1]:
                if pair in edge_map:
                    for i in range(10):
                        yield pair
                else:
                    yield pair
            else:
                if pair in edge_map:
                    print(f'Yo {p1} {p2} {p1-p2}')
    else:
        raise ValueError(f'Relation type {relation_type} is not allowed.')

class TDGPairDataset:
    splits = {
        'train': 'tdg_data/train.txt',
        'dev': 'tdg_data/dev.txt',
        'test': 'tdg_data/test.txt',
    }
    dp_path = Path('tdg_data_dp/dp')

    @classmethod
    def setup_dataset(cls, args, tokenizer):
        dataset = cls(args)
        dataloaders = dataset.load_tdg(tokenizer)
        return dataset, dataloaders

    @property
    def output_size(self):
        return len(self.label2id)

    @property
    def dataset_args(self):
        dataset_args = {'output_size': self.output_size}
        return Namespace(**dataset_args)

    def __init__(self, args):
        self.args = args
        self.label2id = {
            't2t': T2T_LABELS,
            'e2t': E2T_LABELS,
            'e2e': E2E_LABELS,
        }[self.args.relation_type]

    def load_tdg(self, tokenizer):

        dataloaders = {}

        for split_name, file_path in self.splits.items():
            path = self.args.data_path / file_path
            dataloaders[split_name] = self.read_files(path, tokenizer)
        return Namespace(**dataloaders)

    def read_files(self, path, tokenizer):
        with open(path) as open_file:
            articles = open_file.read().strip().split('\n\n\n')

        data = {
            'sents': [], 'spans': [],
            'labels': [], 'label_ids': [],
            'e1_sent_num': [], 'e2_sent_num': [],
        }

        for article in tqdm(articles, disable=self.args.disable_tqdm):
            article_data = self.process_article(article)
            self.process_graph(article_data, data)
        tokenized_data = self.tokenize(data, tokenizer)

        dataset = DictDataset(tokenized_data)

        dataloader = DataLoader(dataset,
            batch_size=self.args.batch_size,
            shuffle=True)
        return dataloader

    def process_article(self, article):
        text, edge_annotations = article.split('EDGE_LIST')
        meta, *sents = text.strip().split('\n')
        doc_meta, paragraph = meta[9:-10].split('<paragraph:')
        paragraph = [int(x) for x in paragraph.split('_')]
        article_id = int(doc_meta.split('"')[1])

        sent_para_map = {-1:-1}
        for para, (start, end) in enumerate(zip([0]+paragraph, paragraph+[len(sents)])):
            for s in range(start, end):
                sent_para_map[s] = para

        # We are using TDG's original tokenization
        sents = [s.split(' ') for s in sents]
        sents.insert(0, 'The publication time of this article .'.split())
        edge_annotations = edge_annotations.strip().split('\n')

        timexs, events = set(), set()
        timexs.add((-1, -1, -1))
        edge_map = {}

        for edge in edge_annotations:
            child, child_type, parent, relation = edge.split('\t')
            child = tuple(int(x) for x in child.split('_'))
            parent = tuple(int(x) for x in parent.split('_'))
            if child_type == 'Timex':
                timexs.add(child)
            elif child_type == 'Event':
                events.add(child)
            edge_map[(child, parent)] = relation

        timexs = sorted(list(timexs))
        events = sorted(list(events))

        article_data = {
            'timexs': timexs,
            'events': events,
            'edge_map': edge_map,
            'sents': sents,
            'article_id': article_id,
            'sent_para_map': sent_para_map,
        }

        return article_data

    def process_graph(self, article_data, data):
        timexs = article_data['timexs']
        events = article_data['events']
        edge_map = article_data['edge_map']
        sents = article_data['sents']
        article_id = article_data['article_id']
        sent_para_map = article_data['sent_para_map']

        for pair in get_pairs(timexs, events, edge_map, sent_para_map, self.args.relation_type):
            (s1, b1, e1), (s2, b2, e2) = pair

            s1 += 1
            s2 += 1

            if (b1, e1) == (-1, -1):
                b1, e1 = 0, -1
            if (b2, e2) == (-1, -1):
                b2, e2 = 0, -1

            if s1 == s2:
                detok_s, detok_spans = detokenize(sents[s1])
                sb1, se1 = detok_spans[b1][0], detok_spans[e1][1]
                sb2, se2 = detok_spans[b2][0], detok_spans[e2][1]
            else:
                detok_s1, detok_spans1 = detokenize(sents[s1])
                detok_s2, detok_spans2 = detokenize(sents[s2])
                detok_s = detok_s1 + ' ' + detok_s2
                s1_length = len(detok_s1 + ' ')
                sb1, se1 = detok_spans1[b1][0], detok_spans1[e1][1]
                sb2, se2 = detok_spans2[b2][0] + s1_length, detok_spans2[e2][1] + s1_length

            data['labels'].append(edge_map.get(pair, 'None'))
            data['sents'].append(detok_s)
            data['spans'].append(((sb1, se1), (sb2, se2)))
            data['e1_sent_num'].append(s1)
            data['e2_sent_num'].append(s2)
            data['label_ids'].append((article_id, s1, b1, e1, s2, b2, e2))

    def tokenize(self, data, tokenizer):
        pt_data = {}

        e1, e2 = zip(*data['spans'])

        tokenized_output = tokenizer(data['sents'], padding=True, return_offsets_mapping=True, return_tensors='pt')

        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        offset_mapping = tokenized_output['offset_mapping']

        ignore = ((offset_mapping[..., 0]==0) & (offset_mapping[..., 1]==0))
        offset_mapping[ignore.unsqueeze(-1).expand(-1,-1,2)] = -1
        e1_offset, e2_offset = torch.LongTensor(e1), torch.LongTensor(e2)

        e1_mask = \
            (offset_mapping[..., 0] >= e1_offset[..., 0].unsqueeze(-1)) & \
            (offset_mapping[..., 1] <= e1_offset[..., 1].unsqueeze(-1))
        e2_mask = \
            (offset_mapping[..., 0] >= e2_offset[..., 0].unsqueeze(-1)) & \
            (offset_mapping[..., 1] <= e2_offset[..., 1].unsqueeze(-1))

        pt_data['input_ids'] = input_ids
        pt_data['attention_mask'] = attention_mask
        pt_data['e1_mask'] = e1_mask
        pt_data['e2_mask'] = e2_mask
        pt_data['labels'] = torch.LongTensor([self.label2id[x] for x in data['labels']])
        pt_data['label_ids'] = torch.LongTensor(data['label_ids'])

        pt_data['e1_sent_num'] = torch.LongTensor(data['e1_sent_num'])
        pt_data['e2_sent_num'] = torch.LongTensor(data['e2_sent_num'])

        return pt_data