from agg_predict import agg_predict
from type_predict import type_predict
from op_predict import op_predict
from value_predict import value_predict
from column_predict import column_predict
from uie.information_extract import get_sentence_value
from transformers import BertTokenizer,BertModel
import json
from tqdm import tqdm
def save_json(data,out_file):
    with open(out_file, "a", encoding="utf-8") as fp:
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
    text=text.strip()
    agg=agg_predict(text,BT,BM)
    value_columns=['产品类别','一级分类','二级分类']
    value_static=['是否海外发售','是否高端产品']
    if agg !="other":
        columns=column_predict(text,agg,BT,BM)

        entities=[]
        print("columns",columns)
        for column in columns:

            agg,column,type,op=predict_Part(text,agg,column,BT,BM)
            if column in value_static:
                value='Y'
            if column in value_columns:
                value=value_predict(text,agg,type,column,BT,BM)
            else:
                value=get_sentence_value(text,column)
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
            text=datas[i]
            e_dict=Predict(text,BT, BM)
            entities.append(e_dict)
            if i==3:
                break


    save_json(entities,out_file)

if __name__ =='__main__':
    BT=BertTokenizer.from_pretrained('bert-base-chinese')
    BM=BertModel.from_pretrained('bert-base-chinese')

    # main('../datasets/traindata-query正文-v1.txt','result_v1.json')
    # main('../datasets/traindata-details-v2.json','result_v2.json')
    i=1
    while(i):
        text=input("请输入文本：")
        save_json(Predict(text,BT, BM),'result1.json')
        i=input("是否继续：")




