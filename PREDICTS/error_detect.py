import json
import difflib

##预测结果正误检测


def get_equal_rate(str1, str2):#计算字符串相似度
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

# 数据路径
path = "../datasets/test_2.json"
path_pre = "result200.json"
# 读取训练集v1

def error_de(filename):
    with open(path, "r",encoding='UTF-8') as f:
        row_data = json.load(f)
        f.close()
    # 预测文件
    with open(path_pre, "r",encoding='UTF-8') as f:
        row_data_pre = json.load(f)
        f.close()

    f = open(filename, 'w',encoding='UTF-8')

    for result in row_data_pre:  # 预测结果
        # print(result)
        less_flag = 0  # 漏检标记
        flag = []
        for data in row_data:  # 训练集
            text = result['text']
            if result['text'] == data['text']:
                if result['agg'] != data['agg']:  # agg
                    print('agg_error:text={text},agg={agg},agg_pre={agg_pre}'.format(text=text,agg=data['agg'],agg_pre=result['agg']),file=f)
                columns = []
                for i in range(len(data['entities'])):
                    columns.append(data['entities'][i]['column'])
                if len(result['entities']) < len(data['entities']):  # entity漏检
                    # print('error-less')
                    less_flag = 1  # 漏检
                for i in range(len(result['entities'])):
                    entity = result['entities'][i]
                    if entity['column'] in columns:
                        flag.append(columns.index(entity['column']))  # 预测正确的的column加入标记
                    elif entity['column'] not in columns:  # column
                        max_column_rate = 0
                        max_column = ''
                        for column in columns:  # 寻找相似度最高的column
                            rate = get_equal_rate(entity['column'], column)
                            if rate > max_column_rate:
                                max_column_rate = rate
                                max_column = column
                        flag.append(columns.index(max_column))  # error column加入标记

                        print('column_error:text={text},column={max_column},column_pre={column_pre}'.format(text=text,max_column=max_column,column_pre=entity['column']),file=f)

                    for entity2 in data['entities']:
                        if entity2['column'] == entity['column']:
                            if entity['type'] != entity2['type']:  # type
                                print('type_error:text={text},type={type},type_pre={type_pre}'.format(text=text,type=entity2['type'],type_pre=entity['type']),file=f)
                            if entity['value'] != entity2['value']:  # value
                                print('value_error:text={text},value={value},value_pre={value_pre}'.format(text=text,value=entity2['value'],value_pre=entity['value']),file=f)
                            if entity['op'] != entity2['op']:  # op
                                print('op_error:text={text},op={op},op_pre={op_pre}'.format(text=text,op=entity2['op'],op_pre=entity['op']),file=f)
                if less_flag:  # entity_less
                    for i in range(len(columns)):
                        if i not in flag:
                            print('entity_less:text={text},column={less_column}'.format(text=text,less_column=columns[i]),file=f)
                    # print(flag)

    f.close()
    print('over')
# error_de("200err.txt")
# str1='123'
# str2='456'
# print(get_equal_rate(str1,str2))