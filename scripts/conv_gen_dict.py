# coding: utf-8
# 从conv格式语料生成dictionary

import os
import Amadeus
import sys

if __name__ == "__main__":
    # argv: 1.a dir, all txt in it; 2.a file, to save the dict; 3. an int, top x words to keep
    inputs = Amadeus.AMADEUS_YULIAO
    output = Amadeus.AMADEUS_DICTIONARY
    topx = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    d = Amadeus.brain.dictionary.Dictionary()
    ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
    for filename in os.listdir(inputs):
        try:
            with open(inputs + os.sep + filename, encoding="utf8") as f:
                for line in Amadeus.data.yuliao_process.preprocess_conv(f):
                    for l in line[0] + line[1]:
                        for word in ws.cut(l.strip()):
                            d[word] = d.get(word, 0) + 1
        except Exception as e:
            print(e)
            continue
        print(filename + " done.")
    print("Wordseg has done.Now sort top " + str(topx))
    topkeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)[:topx]
    # del more than new
    newd = Amadeus.brain.dictionary.Dictionary()
    for i, k in enumerate(topkeys):
        newd[k] = [i, d[k]]
    newd.write_to_file(output)


