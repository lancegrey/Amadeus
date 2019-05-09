# coding: utf-8
import os
import numpy as np
import random
import Amadeus
from Amadeus.brain import dictionary
from Amadeus.data.yuliao_process import tail_name_func
from Amadeus.brain.w2v.w2v_gensim import W2V


def s2v(line, dic, ws, w2v):
    words = []
    # 转换成向量
    for word in ws.cut(line.strip()):
        word_vec = w2v.safe_get(word)
        word_id = dic.get(word, [-1, -1])[0]
        if word_vec is not None:
            words.append([word_vec, word_id])
    return words


def trans_data(inputs, dic, batch_size=8192, debug=False):
    """
    :param inputs: 语料文件
    :param batch_size: 打到一个输出文件的数据量，和train的batch size无关
    :param debug: bool
    :return:
    """
    w2v = W2V()
    ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
    data = {"data": [], "label": []}
    if debug:
        data["debug"] = []
    batch_counter = 0
    for filename in os.listdir(inputs):
        with open(inputs + os.sep + filename, encoding="utf8") as f:
            tail_name = filename.split(".")[-1]
            processor = tail_name_func.get(tail_name, Amadeus.data.yuliao_process.preprocess_unknown)
            for conversation in processor(f):
                if len(conversation) < 2:
                    continue
                last_sentence = []
                last_line = ""
                for line in conversation:
                    vectors = s2v(line, dic, ws, w2v)
                    if len(vectors) <= 0:
                        last_sentence = []
                        last_line = ""
                    if len(last_sentence) > 0:
                        data["data"].append([x for x in last_sentence])
                        data["label"].append([x for x in vectors])
                        if debug:
                            data["debug"].append([last_line, line])
                        batch_counter += 1
                        if batch_counter >= batch_size - 1:
                            shuffle = list(zip(data["data"], data["label"]))
                            random.shuffle(shuffle)
                            data = {"data": [x[0] for x in shuffle], "label": [x[1] for x in shuffle]}
                            yield data
                            data = {"data": [], "label": []}
                            if debug:
                                data["debug"] = []
                            batch_counter = 0
                    last_sentence = vectors
                    last_line = line

            # 填不满batch的最后几个数据
            if len(data) > 0:
                yield data


if __name__ == "__main__":
    inputs = Amadeus.AMADEUS_YULIAO
    DIC = dictionary.Dictionary()
    DIC.read_from_file(Amadeus.AMADEUS_DICTIONARY)
    outs = trans_data(inputs, DIC, debug=False)
    for i, data in enumerate(outs):
        print("batch:", i)
        np.save(Amadeus.AMADEUS_TRAIN_DATA_DIR + os.sep + str(i), data)
        # for row in zip(data["data"], data["label"], data["debug"]):
        #     d, l, s = row
        #     print(s[0], s[1])
