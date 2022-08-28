import os
import json
import logging
import codecs
import os
import argparse

from TRAINS.op_train import op_train
from TRAINS.type_train import type_train
from TRAINS.agg_train import agg_train
from TRAINS.value_train import value_train
from TRAINS.column_train import column_train


from PREDICTS.error_detect import error_de

from transformers import BertTokenizer, BertModel

logging.basicConfig(level=logging.ERROR)
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))

def data_split(file_path,out_path):
    print("Loading Data…………………………")

    val_r=0.3

    with open(file_path,'r',encoding='utf-8') as file:
        datas=json.load(file)
        for data in datas:
            entities=data['entities']
            for e in entities:
                if e['column']=="是否高端产品":
                    e['value']="Y"
        datas_len=len(datas)
        test_l=200
        test=datas[0:test_l]
        with codecs.open(out_path+'test_2.json','w', 'utf-8') as outf:
            json.dump(test, outf, ensure_ascii=False)
            outf.write('\n')
            outf.close()
        with open(out_path+'test_2.json','r',encoding='utf-8') as file:
            fp=open(out_path+'query-test.txt','w',encoding='utf-8')
            datas_test=json.load(file)
            for data in datas_test:
                str1=data['text']+'\n'
                fp.write(str1)

        val_l=int(datas_len*val_r)
        train_l=datas_len-val_l-test_l
        val=datas[test_l:test_l+val_l]
        train=datas[test_l+val_l:]


        for data in datas:
            if data['agg']==None:
                data['agg']="other"
            entities=data['entities']
            for e in entities:
                if e['op']==None:
                    e['op']="other"
                if e['op']=="not in":
                    e['op']="not_in"
                if e['column']=="近五次评级":
                    e['column']="近5次评级"
                if e['column']=="近五年评级":
                    e['column']="近5次评级"
                if e['column']=="近三年评级":
                    e['column']="近三次评级"







        print("train_size:{}, val_size:{}, test_size:{}".format(train_l,val_l,test_l))



    with codecs.open(out_path+'test.json','w', 'utf-8') as outf:
        json.dump(test, outf, ensure_ascii=False)
        outf.write('\n')
        outf.close()

    with codecs.open(out_path+'val.json','w', 'utf-8') as outf:
        json.dump(val, outf, ensure_ascii=False)
        outf.write('\n')
        outf.close()

    with codecs.open(out_path+'train.json','w', 'utf-8') as outf:
        json.dump(train, outf, ensure_ascii=False)
        outf.write('\n')
        outf.close()



if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='../pretrained_models/bert-base-chinese')
    parser.add_argument("--data_path", type=str,default='../datasets/traindata-details-v1.json')
    parser.add_argument("--data_split_path", type=str, default='../datasets/')
    hparams = parser.parse_args()

    BT=BertTokenizer.from_pretrained(hparams.pretrained_model_path)
    BM=BertModel.from_pretrained(hparams.pretrained_model_path)


    data_split(hparams.data_path,hparams.data_split_path)


    print("Agg Training…………………………")
    agg_train(BT,BM)
    print("Type Training…………………………")
    type_train(BT,BM)
    print("Op Training…………………………")
    op_train(BT,BM)
    print("Column Training…………………………")
    column_train(BT,BM)
    print("value Training…………………………")
    value_train(BT,BM)




