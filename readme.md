## 赛题：面向口语的NL2SQL语义解析算法
### 1、安装requirements中的包
### 2、数据集相关
对于数据集中列名与给的excel表格不符合的情况，本文以excel为准。
（例如：数据集中的column=“近五年评级”、“近五次评级”都处理为Excel中的“近5次评级”来进行训练）模型预测的结果为“近5次评级”

### 3、预测过程
    运行PREDICT/demo_predict.py 输入和输出的文件路径在main函数中更改
    默认为input_file='../datasets/traindata-query正文-v1.txt'
    out_file='result.json'
### 4、训练过程
    运行TRAIN/demo_train.py
    其中data_path为数据集路径，即'traindata-details-v1.json'所在
    data_split_path为数据集分割后的路径，取前200条为测试集，剩余作为训练集和验证集。训练集:验证集=7:3
#### 本模型训练分为五个部分：agg,column,type,value,op分开训练，其参数调节可在对应文件夹中修改
