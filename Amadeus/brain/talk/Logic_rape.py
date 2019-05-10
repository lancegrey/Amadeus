# coding: utf-8
import Amadeus
import json
import numpy as np
import re


class LogicRape(object):
    def __init__(self):
        with open(Amadeus.LOGIC_RAPE_DIC) as f:
            dic = json.loads(f.read())
        self.re_dic = {}
        self.re_list = []
        i = 0
        for k, v in dic.items():
            if v["type"] == "str":
                k = "(^" + self.pre_format(k) + "$)"
                self.re_dic[i] = v["answers"]
            self.re_list.append(k)
            i += 1
        for k, v in dic.items():  # 优先re
            if v["type"] == "re":
                k = "(" + k + ")"
                self.re_dic[i] = v["answers"]
                self.re_list.append(k)

        # print("|".join(self.re_list))
        self.re_str = re.compile("|".join(self.re_list))

    def pre_format(self, k):
        k = k.replace("?", "\?")
        k = k.replace("\\", "\\\\")
        k = k.replace(".", "\.")
        k = k.replace("!", "\!")
        k = k.replace("(", "\(")
        k = k.replace(")", "\)")
        k = k.replace("*", "\*")
        k = k.replace("^", "\^")
        k = k.replace("$", "\$")
        k = k.replace("[", "\[")
        k = k.replace("]", "\]")
        k = k.replace("&", "\&")
        return k

    def search(self, query):
        answers = []
        ret = self.re_str.findall(query)
        # print(ret)
        for r in ret:
            for i, rr in enumerate(r):
                if len(rr) > 0:
                    answers += self.re_dic[i]
        return answers


if __name__ == "__main__":
    LR = LogicRape()
    print(LR.search("你喜欢吃什么"))
