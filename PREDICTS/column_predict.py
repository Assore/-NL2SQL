import os

from column.predict import predict
from column.hparams import hparams
from transformers import BertTokenizer,BertModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def column_predict(text,agg,BT,BM):
    column=predict(hparams,text,agg,BT,BM)
    return column

if __name__=='__main__':
    BT=BertTokenizer.from_pretrained('bert-base-chinese')
    BM=BertModel.from_pretrained('bert-base-chinese')

    i=1
    while(i):
        text=input("text：")
        agg=input("agg：")
        print(column_predict(text,agg,BT,BM))
        i=input("是否继续")
