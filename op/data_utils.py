import re
import os
import json

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))





def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(filename,tokenizer=None,max_len=128):
    token_list=[]
    tags=[]
    with open(filename,'r',encoding='utf-8') as file:
        datas=json.load(file)
        for data in datas:
            text=data['text']
            agg=data['agg']
            entities=data['entities']

            for e in entities:
                type=e['type']
                op=e['op']
                column=e['column']
                text1=text+'['+agg+']'+'['+type+']'+'['+column+']'
                tokens=tokenizer.tokenize(text1)
                token_list.append(tokens)
                tags.append(op)
    return token_list,  tags


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class SentenceREDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128,Tokenizer=None):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = Tokenizer
        self.max_len = max_len
        self.tokens_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)
        self.tag2idx = get_tag2idx(self.tagset_path)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_tag = self.tags[idx]
        encoded = self.tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }
        return sample
