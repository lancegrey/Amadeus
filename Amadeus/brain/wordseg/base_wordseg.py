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



