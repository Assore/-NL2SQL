import os
import re
import torch
import numpy as np
from column.data_utils import  get_idx2tag,convert_pos_to_mask

from column.model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))

def get_Tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))
def predict(hparams,text,agg,BT,BM):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file

    idx2tag = get_idx2tag(tagset_file)
    Tags=get_Tag(tagset_file)
    hparams.tagset_size = len(idx2tag[0])
    model = SentenceRE(hparams,BM).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    tokenizer = BT

    text1=text+'['+agg+']'
    tokens=tokenizer.tokenize(text1)
    encoded = tokenizer.encode_plus(tokens, max_length=hparams.max_len, pad_to_max_length=True)
    input_ids = torch.tensor(encoded['input_ids']).reshape(1,128).to(device)
    token_type_ids = torch.tensor(encoded['token_type_ids']).reshape(1,128).to(device)
    attention_mask = torch.tensor(encoded['attention_mask']).reshape(1,128).to(device)

    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)[0]
        logits = logits.to(torch.device('cpu'))
        pre=np.array(logits).round().reshape(-1)
        col=[]
        for p in range(len(pre)):
            if pre[p]==1:
                col.append(Tags[p])
        #logits是模型的输出，表示输入的句子属于三个类的概率，argmax表示取最大值的索引，idx2tag则该索引对应的tag是什么
    return col


