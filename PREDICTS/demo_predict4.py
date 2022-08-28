from agg_predict import agg_predict
from type_predict import type_predict
from op_predict import op_predict
from column_predict import column_predict
from value_predict import value_predict
from uie.information_extract import get_sentence_value_2
from transformers import BertTokenizer,BertModel
import json
from tqdm import tqdm
import os
import argparse
import codecs
import random
from error_detect import error_de
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
    value_columns=['产品类别','一级分类','二级分类']

    agg=agg_predict(text,BT,BM)
    if agg !="other":
        columns=column_predict(text,agg,BT,BM)

        entities=[]
        values=get_sentence_value_2(text,columns)[0]
        for column in columns:

            agg,column,type,op=predict_Part(text,agg,column,BT,BM)
            if type=="condition":
                if column in value_columns:
                    value=value_predict(text,agg,type,column,BT,BM)
                else:
                    if column in values.keys():
                        value=values[column][0]['text']
                    else:
                        value="other"
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
            # if i==3:
            #     break


    save_json(entities,out_file)

def data_split(file_path,out_path):
    # print("Loading Data…………………………")
    #
    #
    # with open(file_path,'r',encoding='utf-8') as file:
    #     datas=json.load(file)
    #
    #     for data in datas:
    #
    #         entities=data['entities']
    #         for e in entities:
    #
    #             if e['column']=="近五次评级":
    #                 e['column']="近5次评级"
    #             if e['column']=="近五年评级":
    #                 e['column']="近5次评级"
    #             if e['column']=="近三年评级":
    #                 e['column']="近三次评级"
    #
    #
    #
    #
    #
    #     test=datas[0:11]
    #
    #
    #
    # with codecs.open(out_path+'test.json','w', 'utf-8') as outf:
    #     json.dump(test, outf, ensure_ascii=False)
    #     outf.write('\n')
    #     outf.close()

    with open(out_path+'test.json','r',encoding='utf-8') as file:
        fp=open(out_path+'query-test.txt','w',encoding='utf-8')
        datas=json.load(file)
        for data in datas:
            str1=data['text']+'\n'
            fp.write(str1)

if __name__ =='__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='../pretrained_models/bert-base-chinese')
    parser.add_argument("--data_path", type=str, default='../datasets/traindata-details-v2.json')
    parser.add_argument("--data_split_path", type=str, default='../datasets/')
    parser.add_argument("--input_file", type=str, default='../datasets/query-test.txt')
    parser.add_argument("--output_file", type=str, default='result.json')
    parser.add_argument("--err_file", type=str, default='../datasets/error')
    hparams = parser.parse_args()


    BT=BertTokenizer.from_pretrained(hparams.pretrained_model_path)
    BM=BertModel.from_pretrained(hparams.pretrained_model_path)
    data_split(hparams.data_path, hparams.data_split_path)
    main(hparams.input_file,hparams.output_file)
    error_de(hparams.err_file+'.txt')
    #main的两个参数，前者为输入文件路径，后者为输出文件路径







