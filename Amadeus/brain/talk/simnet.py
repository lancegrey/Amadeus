import tensorflow as tf
import numpy as np


class SimNet(object):
    def __init__(self):
        self.query_input = tf.placeholder(tf.int32, shape=[None, None], name='query_input')
        self.ans_input_pos = tf.placeholder(tf.int32, shape=[None, None], name='query_pos')
        self.x = 0
