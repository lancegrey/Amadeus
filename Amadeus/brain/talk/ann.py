# coding: utf-8
import sys
import annoy
import os
import numpy as np
import random
import Amadeus
from Amadeus.brain import dictionary
from Amadeus.data.yuliao_process import tail_name_func
from Amadeus.brain.w2v.w2v_gensim import W2V


def s2v1(line, ws, w2v):
    words = np.zeros(256, np.float32)
    # 转换成向量
    i = 0
    for word in ws.cut(line.strip()):
        word_vec = w2v.safe_get(word)
        # word_id = dic.get(word, [-1, -1])[0]
        if word_vec is not None:
            i += 1
            words += word_vec
    return words / (i if i > 0 else 1)


def trans_data(inputs):
    """
    :param inputs: 语料文件
    :param batch_size: 打到一个输出文件的数据量，和train的batch size无关
    :param debug: bool
    :return:
    """
    w2v = W2V()
    ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
    # ws = Amadeus.brain.wordseg.base_wordseg.ZSeg()
    data = {}
    i = 0
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
                    vectors = s2v1(line, ws, w2v)
                    if len(vectors) <= 0:
                        last_sentence = []
                        last_line = ""
                    if len(last_sentence) > 0:
                        if last_line not in data:
                            data[last_line] = [last_sentence]
                        data[last_line].append(line)
                        i += 1
                        if i % 1000 == 0:
                            print(i)
                            #i = 0
                    last_line = line
                    last_sentence = vectors
    return data


def build_dict():
    print("build dict")
    inputs = Amadeus.AMADEUS_YULIAO
    DIC = dictionary.Dictionary()
    DIC.read_from_file(Amadeus.AMADEUS_DICTIONARY)
    outs = trans_data(inputs)
    np.save(Amadeus.ANNOY_DICT, outs)
    print("dict done")


def buid_tree():
    print("build tree")
    outs = np.load(Amadeus.ANNOY_DICT).item()
    tree = annoy.AnnoyIndex(256)
    keys = sorted(outs.keys())
    np.save(Amadeus.ANNOY_DICT_KEYS, keys)
    for i, k in enumerate(keys):
        v = outs[k][0]
        tree.add_item(i, v)
    tree.build(Amadeus.ANNOY_TREE_NUM)
    tree.save(Amadeus.ANNOY_TREE)
    print("tree done")


class AnnoySearch(object):
    def __init__(self):
        self.tree = annoy.AnnoyIndex(Amadeus.ANNOY_WORD_EMBADDING_NUM)
        self.tree.load(Amadeus.ANNOY_TREE)
        self.dic_keys = np.load(Amadeus.ANNOY_DICT_KEYS)
        self.ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
        self.w2v = W2V()
        self.dic = np.load(Amadeus.ANNOY_DICT).item()

    def search(self, query):
        vec = AnnoySearch.s2v1(query, self.ws, self.w2v)
        similar = self.tree.get_nns_by_vector(vec, 1, include_distances=True)
        similar_id, similar_distance = similar[0][0], similar[1][0]
        similar_query = self.dic_keys[similar_id]
        answers = self.dic[similar_query][1:]
        return similar_query, similar_distance, answers

    @staticmethod
    def s2v1(line, ws, w2v):
        words = np.zeros(256, np.float32)
        # 转换成向量
        i = 0
        for word in ws.cut(line.strip()):
            word_vec = w2v.safe_get(word)
            # word_id = dic.get(word, [-1, -1])[0]
            if word_vec is not None:
                i += 1
                words += word_vec
        return words / (i if i > 0 else 1)


if __name__ == "__main__":
    #build_dict()
    #buid_tree()
    AS = AnnoySearch()
    test = ["我喜欢达达", "你叫什么名字", "我是白熊", "天气好吗", "hi", "Amadeus"]
    for k in test:
        ret = AS.search(k)
        print(k, ret[0], ret[1])
        for v in ret[2]:
            print(v)

