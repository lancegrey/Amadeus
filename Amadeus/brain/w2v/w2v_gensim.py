# coding: utf-8
# word to vector by gensim

import gensim
import numpy as np
import Amadeus


class W2V(object):
    def __init__(self):
        model = gensim.models.Word2Vec.load(Amadeus.AMADEUS_GENSIM_MODEL)
        self.kv = model.wv

    def __getitem__(self, item):
        return self.kv[item]

    def safe_get(self, item):
        if item not in self.kv:
            return None
        else:
            return self[item]

    def cos(self, x, y):
        x = self[x]
        y = self[y]
        up = np.dot(x, y)
        down = sum(x**2)**0.5 * sum(y**2)**0.5
        return up/down


if __name__ == "__main__":
    w2v = W2V()
    test = [["qq", "百度"],
            ["qq", "腾讯"],
            ["手机", "苹果"],
            ["手机", "梨"],
            ["男人", "男子"],
            ["男人", "玻璃"],
            ["爱", "喜欢"],
            ["爱", "死亡"],
            ["AI", "机器学习"],
            ["AI", "连续剧"]]

    for t in test:
        print(*t, w2v.cos(*t))
        # print("cos: ", w2v.cos(*t))

