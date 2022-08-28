import os

from column.train import train
from column.hparams import hparams
from transformers import BertTokenizer,BertModel
import logging
logging.basicConfig(level=logging.ERROR)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def column_train(BT,BM):
    train(hparams,BT,BM)

if __name__ =='__main__':
    BT=BertTokenizer.from_pretrained('bert-base-chinese')
    BM=BertModel.from_pretrained('bert-base-chinese')
    column_train(BT,BM)
