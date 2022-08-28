import copy
import json

from tqdm import trange

d = ["有什么产品", "有哪些产品"]


def augment(sample: dict, json_obj: list):
    global nxt_id
    text = sample["text"]
    for word in d:
        if word in text:
            new_sample = copy.deepcopy(sample)
            new_sample["text"] = new_sample["text"].replace(word, "")
            for i in range(len(new_sample["entities"])):
                entity = new_sample["entities"][i]
                if entity["text"] != word:
                    continue
                entity["text"] = ""
                entity["start"] = 0
                entity["end"] = 0
            new_sample["id"] = nxt_id
            nxt_id += 1
            json_obj.append(new_sample)


json_obj = json.load(open("traindata-details-v2.json", "r", encoding="utf8"))
L = len(json_obj)
print(len(json_obj))
nxt_id = len(json_obj) + 1
for i in trange(L):
    augment(json_obj[i], json_obj)
print(len(json_obj))
json.dump(json_obj, open("traindata-details-v3.json", "w", encoding="utf8"),ensure_ascii=False)
