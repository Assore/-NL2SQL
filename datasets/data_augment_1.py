import copy
import json

from tqdm import trange

table = {
    "产品类别": ["通讯设备", "办公设备", "穿戴设备"],
    "一级分类": ["手机", "PC", "穿戴"],
    "二级分类": ["海外定制版", "高端手机", "中端手机", "M系列笔记本", "台式机", "平板电脑", "智能手表", "家用医疗", "智能耳机", "智能眼镜"],
    "最近一次评级": ["5", "4", "3", "2", "1"]
}
d = ["有什么产品"]


def augment(sample: dict, json_obj: list):
    global nxt_id
    text = sample["text"]
    for i in range(len(sample["entities"])):
        entity = sample["entities"][i]
        if entity["type"] == "condition" and entity["column"] in table.keys():
            value = entity["value"]
            for new_value in table[entity["column"]]:
                if new_value == value:
                    continue
                pos = text.find(value)
                if pos == -1:
                    continue
                new_sample = copy.deepcopy(sample)
                new_sample["text"] = new_sample["text"].replace(value, new_value)
                new_sample["entities"][i]["text"] = new_sample["entities"][i]["text"].replace(value, new_value)
                new_sample["entities"][i]["end"] += len(new_value) - len(value)
                new_sample["entities"][i]["value"] = new_value
                for j in range(i + 1, len(sample["entities"])):
                    new_sample["entities"][j]["start"] += len(new_value) - len(value)
                    new_sample["entities"][j]["end"] += len(new_value) - len(value)
                new_sample["id"] = nxt_id
                nxt_id += 1
                json_obj.append(new_sample)


json_obj = json.load(open("traindata-details-v1.json", "r", encoding="utf8"))
L = len(json_obj)
print(len(json_obj))
nxt_id = len(json_obj) + 1
for i in trange(L):
    augment(json_obj[i], json_obj)
print(len(json_obj))
json.dump(json_obj, open("traindata-details-v2.json", "w", encoding="utf8"),
          ensure_ascii=False)
