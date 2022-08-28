import os

from agg.predict import predict
from agg.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def agg_predict(text,BT,BM):
    agg=predict(hparams,text,BT,BM)
    return agg



