import json
import os
import Amadeus


def add_dict_to_conf(add_dict):
    js_dic = {}
    if os.path.exists(Amadeus.LOGIC_RAPE_DIC):
        with open(Amadeus.LOGIC_RAPE_DIC) as f:
            js_dic = json.loads(f.read())
            print(len(js_dic))

    for k, v in add_dict.items():
        js_dic[k] = v

    with open(Amadeus.LOGIC_RAPE_DIC, "w") as f:
        f.write(json.dumps(js_dic))


def add_ini_to_dict():
    path = "E:\PySpace\Amadeus\Amadeus\data\open\\raw_chat_corpus\\raw_chat_corpus\chatterbot-1k\chinese"
    add_dic = {}
    for filename in os.listdir(path):
        truefile = path + os.sep + filename
        with open(truefile, encoding="utf-8") as f:
            data = f.readlines()[3:]
            ques = [data[i][4:].strip() for i in range(len(data)) if i % 2 == 0]
            ans = [data[i][4:].strip() for i in range(len(data)) if i % 2 == 1]
            for q, a in zip(ques, ans):
                if q in add_dic:
                    add_dic[q]["answers"].append(a)
                else:
                    add_dic[q] = {"type": "str", "answers": [a]}
    add_dict_to_conf(add_dic)


add_ini_to_dict()
