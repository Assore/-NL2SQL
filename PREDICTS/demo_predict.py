from agg_predict import agg_predict
from type_predict import type_predict
from op_predict import op_predict
from column_predict import column_predict
from value_predict import value_predict
from uie.information_extract import get_sentence_value
from transformers import BertTokenizer,BertModel
import json
from tqdm import tqdm
import os
import argparse

def save_json(data,out_file):
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

def predict_Part(text,agg,column,BT,BM):

    type=type_predict(text,agg,column,BT,BM)
    op=op_predict(text,agg,type,column,BT,BM)


    return agg,column,type,op

def Predict(text,BT,BM):
    value_columns=['产品类别','一级分类','二级分类']
    values_sta=['是否海外发售','是否高端产品']
    agg=agg_predict(text,BT,BM)
    if agg !="other":
        columns=column_predict(text,agg,BT,BM)

        entities=[]
        for column in columns:

            agg,column,type,op=predict_Part(text,agg,column,BT,BM)
            if type=="condition":
                if column in values_sta:
                    value="Y"
                elif column in value_columns:
                    value=value_predict(text,agg,type,column,BT,BM)
                else:
                    value=get_sentence_value(text,column)
            else:
                value="other"
            if value=="other":
                value=None
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





