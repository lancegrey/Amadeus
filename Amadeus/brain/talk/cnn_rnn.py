# coding: utf-8
# This is the base model of Amadeus
# embadding-conv-pool-rnn

import tensorflow as tf
import numpy as np


class Model(object):
    """
    model
    """
    def __init__(self):
        self.embadding_num = 256
        with tf.variable_scope("inputs_embadding_CNN"):
            # load w2v nn
            pass
            # batch, sentence len, embadding size
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.embadding_num])
            self.


