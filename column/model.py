import os
import logging
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

here = os.path.dirname(os.path.abspath(__file__))


class SentenceRE(nn.Module):

    def __init__(self, hparams,BM):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        self.bert_model = BM

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 2)
        self.hidden2tag = nn.Linear(self.embedding_dim * 2, self.tagset_size)
        self.sig=nn.Sigmoid
    def forward(self, token_ids, token_type_ids, attention_mask):
        #调用bert，其中sequence_out是bert最后一层的输出，pooled_output代表整个句子
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
        # 每个实体的所有token向量的平均值
        e1_h = torch.mean(sequence_output,dim=1)
        e1_h = self.activation(self.dense(e1_h))
        concat_h = torch.cat([pooled_output, e1_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))
        logits=nn.functional.sigmoid(logits)
        return logits


