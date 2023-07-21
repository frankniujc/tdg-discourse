from transformers import BertPreTrainedModel, AutoModel
from transformers import RobertaConfig, RobertaModel

import torch
import torch.nn as nn

from tqdm import tqdm

from ..modules import Classifier, PositionalEmbedding

class TDGBaseline(BertPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config=None, args=None, output_size=None, **_) -> None:
        super().__init__(config)

        self.args = args
        self.roberta = RobertaModel.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = Classifier(
            hidden_size=config.hidden_size * 2,
            output_size=output_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, **_):
        embeddings = self.roberta(input_ids, attention_mask=attention_mask)[0]
        e1_embedding = (embeddings * e1_mask.unsqueeze(-1)).mean(dim=1)
        e2_embedding = (embeddings * e2_mask.unsqueeze(-1)).mean(dim=1)
        event_pair = torch.cat([e1_embedding.squeeze(dim=1), e2_embedding.squeeze(dim=1)], dim=1)
        logits = self.classifier(event_pair.unsqueeze(dim=1)).squeeze()
        return logits

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class TDGAbsolutePositionEncoding(TDGBaseline):

    def __init__(self, config=None, args=None, output_size=None):
        super().__init__(config=config, args=args,
            output_size=output_size)
        self.sentence_position_embedding = PositionalEmbedding(config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None,
        e1_mask=None, e2_mask=None, e1_sent_num=None, e2_sent_num=None,
        **_):

        embeddings = self.roberta(input_ids, attention_mask=attention_mask)[0]
        e1_embedding = (embeddings * e1_mask.unsqueeze(-1)).sum(dim=1)
        e2_embedding = (embeddings * e2_mask.unsqueeze(-1)).sum(dim=1)

        e1_s_pos_emb = self.sentence_position_embedding(e1_sent_num).squeeze()
        e2_s_pos_emb = self.sentence_position_embedding(e2_sent_num).squeeze()

        e1_embedding = e1_embedding + e1_s_pos_emb
        e2_embedding = e2_embedding + e2_s_pos_emb

        event_pair = torch.cat([e1_embedding, e2_embedding], dim=1)

        logits = self.classifier(event_pair.unsqueeze(dim=1)).squeeze()
        return logits

def train_epoch(self, epoch):
    self.model.train()
    if self.args.multi_gpus:
        self.dataloaders.train.sampler.set_epoch(epoch)

    with tqdm(self.dataloaders.train, disable=self.args.disable_tqdm) as pbar:
        for i, batch in enumerate(pbar):
            batch = self.process_batch(batch)
            labels = batch.pop('labels')

            outputs = self.model(**batch)
            loss = self.loss_function(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(loss=loss.item())

def setup_training(self):
    optimizer_cls = getattr(torch.optim, self.args.optimizer)
    self.optimizer = optimizer_cls(self.model.parameters(), lr=self.args.learning_rate)

tdg_baseline_trainloop = TDGBaseline, train_epoch, setup_training
tdg_spe_trainloop = TDGAbsolutePositionEncoding, train_epoch, setup_training