import os

from type.predict import predict
from type.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def type_predict(text,agg,column,BT,BM):
    type=predict(hparams,text,agg,column,BT,BM)
    return type

