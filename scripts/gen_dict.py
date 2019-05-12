# coding: utf-8
# 生成dictionary

import os
import Amadeus
import sys


def form_yuliao():
    inputs = Amadeus.AMADEUS_YULIAO
    output = Amadeus.AMADEUS_DICTIONARY
    topx = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    d = Amadeus.brain.dictionary.Dictionary()
    ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
    #ws = Amadeus.brain.wordseg.base_wordseg.ZSeg()
    for filename in os.listdir(inputs):
        print(filename + " start.")
        try:
            with open(inputs + os.sep + filename, encoding="utf8") as f:
                tail_name = filename.split(".")[-1]
                processor = Amadeus.data.yuliao_process.tail_name_func.get(tail_name,
                                                                           Amadeus.data.yuliao_process.preprocess_unknown)
                for lines in processor(f):
                    for l in lines:
                        for word in ws.cut(l.strip()):
                            d[word] = d.get(word, 0) + 1
        except Exception as e:
            print(e)
            continue
        print(filename + " done.")
    print("Wordseg has done.Now sort top " + str(topx))
    topkeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)[:topx]
    print(topkeys)
    print(len(d.keys()), len(topkeys))
    # del more than new
    newd = Amadeus.brain.dictionary.Dictionary()
    for i, k in enumerate(topkeys):
        newd[k] = [i, d[k]]
    newd.write_to_file(output)


def form_jb_dic():
    d = Amadeus.brain.dictionary.Dictionary()
    topx = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    output = Amadeus.AMADEUS_DICTIONARY
    with open(Amadeus.AMADEUS_JBDICT, encoding="utf8") as f:
        for line in f:
            word, tf, e = line.strip().split(" ")
            d[word] = int(tf)
    topkeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)[:topx]
    print(len(d.keys()), len(topkeys))
    print("\n".join(topkeys[0:100]))
    # del more than new
    newd = Amadeus.brain.dictionary.Dictionary()
    for i, k in enumerate(topkeys):
        newd[k] = [i, d[k]]
    newd.write_to_file(output)


if __name__ == "__main__":
    #form_jb_dic()
    form_yuliao()
