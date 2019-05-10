# coding: utf-8
# Using python pakages

import jieba
import numpy


class WordSeg(object):
    def __init__(self):
        pass

    def cut(self, text, **kargs):
        return self._cut(text, kargs)

    def _cut(self, text, kargs):
        return ""


class JiebaSeg(WordSeg):
    def __init__(self):
        super(JiebaSeg, self).__init__()

    def _cut(self, text, kargs):
        return jieba.cut(text)  # a generater


class ZSeg(WordSeg):
    def __init__(self):
        super(ZSeg, self).__init__()

    def _cut(self, text, kargs):
        return list(text)


if __name__ == "__main__":
    zseg = ZSeg()
    for w in zseg.cut("白熊吃了一只海豹"):
        print(w)
