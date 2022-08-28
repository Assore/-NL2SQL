from agg_predict import agg_predict
from type_predict import type_predict
from op_predict import op_predict
from column_predict import column_predict
from uie.information_extract import get_sentence_value
from transformers import BertTokenizer,BertModel
import json
from tqdm import tqdm
import os
import argparse

def save_json(data,out_file):
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
        # data：字典的数据
        # fp：保存的文本
        # encoding="utf-8"：使中文能够显示出来，不至于乱码
        # ensure_ascii：保证了 “篮球” 能正确的写入，而不是字节形式
        # indent=4：为了美观，不然会保存成一行

def predict_Part(text,agg,column,BT,BM):

    type=type_predict(text,agg,column,BT,BM)
    op=op_predict(text,agg,type,column,BT,BM)


    return agg,column,type,op

def Predict(text,BT,BM):

    agg=agg_predict(text,BT,BM)
    if agg !="other":
        columns=column_predict(text,agg,BT,BM)

        entities=[]
        for column in columns:
            value=get_sentence_value(text,column)
            if value=="other":
                value=None
            agg,column,type,op=predict_Part(text,agg,column,BT,BM)
            if op=="other":
                op=None
            if op=="not_in":
                op="not in"
            e=dict(type=type,column=column,value=value,op=op)
            entities.append(e)
        e_dict=dict(text=text,agg=agg,entities=entities)

        # print(predict(text,agg,column,BT,BM))
        return e_dict
    else:
        agg=None
        e_dict=dict(text=text,agg=agg,entities=[])
        return e_dict

def main(input_file,out_file):
    entities=[]
    with open(input_file,'r',encoding='utf-8') as file:
        datas=file.readlines()
        for i in tqdm(range(len(datas))):
            data=datas[i]
            text=data.strip()
            e_dict=Predict(text,BT, BM)
            entities.append(e_dict)
            if i==3:
                break


    save_json(entities,out_file)

if __name__ =='__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='../pretrained_models/bert-base-chinese')
    parser.add_argument("--input_file", type=str, default='../datasets/query-test.txt')
    parser.add_argument("--output_file", type=str, default='result.json')
    hparams = parser.parse_args()

    BT=BertTokenizer.from_pretrained(hparams.pretrained_model_path)
    BM=BertModel.from_pretrained(hparams.pretrained_model_path)
    #main的两个参数，前者为输入文件路径，后者为输出文件路径
    main(hparams.input_file,hparams.output_file)





