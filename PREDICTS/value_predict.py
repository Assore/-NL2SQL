import os

from value.predict import predict
from value.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def value_predict(text,agg,type,column,BT,BM):
    value=predict(hparams,text,agg,type,column,BT,BM)
    return value


