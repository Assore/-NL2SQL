import os

from op.predict import predict
from op.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def op_predict(text,agg,type,column,BT,BM):
    op=predict(hparams,text,agg,type,column,BT,BM)
    return op


