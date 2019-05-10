# coding: utf-8
# prepreocess conv

import sys
import csv


def preprocess_conv(f):
    ret = []
    for line in f.readlines():
        if line.strip() != "E":
            ret.append(line[1:].strip())
        if line.strip() == "E" and len(ret) > 0:
            # yield [[x for i, x in enumerate(ret) if i%2 == 0], [x for i, x in enumerate(ret) if i%2 == 1]]
            yield ret
            ret = []

    # E as an ending
    # pass


def preprocess_qingyun(f):
    for line in f.readlines():
        conv = line.strip().split("|")
        conv = [x.strip() for x in conv]
        yield conv


def preprocess_qa(f):
    ret = []
    for line in f:
        q, a = line.strip().split("\t")
        if len(ret) == 0:
            ret.append(q)
            ret.append(a)
        else:
            if ret[-1] == q:
                ret.append(a)
            else:
                yield ret
                ret = []

def preprocess_unknown(f):
    ret = []
    i = 0
    for line in f:
        ret.append(line.strip())
        if i % 2 != 0:
            yield ret
            ret = []
            i = 0  # 防止溢出
        i += 1


tail_name_func = {
    "conv": preprocess_conv,
    "qa": preprocess_qa,
    "qingyun": preprocess_qingyun
}


if __name__ == "__main__":
    f = open("E:\PySpace\Amadeus\Amadeus\data\yuliao\qingyun.qingyun", encoding="utf-8")
    for x in preprocess_qingyun(f):
        print(x)
