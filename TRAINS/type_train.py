import os

from type.train import train
from type.hparams import hparams

import logging
logging.basicConfig(level=logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def type_train(BT,BM):
    train(hparams,BT,BM)


if __name__ == '__main__':
    main()
