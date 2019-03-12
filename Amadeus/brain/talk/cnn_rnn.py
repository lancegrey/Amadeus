# coding: utf-8
# This is the base model of Amadeus
# embadding-conv-pool-rnn

import tensorflow as tf
import numpy as np
from functools import reduce


class Model(object):
    """
    model
    """
    def __init__(self, embadding_num=256):
        self.embadding_num = embadding_num
        # if sentence len is smaller than k, it's hard to use max-k pooling without padding;so k=1;
        # padding the sentence with zeros-vector before input to ensure that 
        # the length of sentence is biger than all of filter sizes
        self.max_k = 1  
        self.cnn_layer_w_para = [[1, 32], [2, 32], [3, 32], [4, 32], [5, 32]]  # filter size, filter num
        self.dense_hidden_nums = [256]
        with tf.variable_scope("inputs_embadding_CNN_dense"):
            # load w2v nn
            pass
            # add position info
            pass
            # batch, sentence len, embadding size
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.embadding_num, 1])
            # conv layer
            self.cnn_outs = []
            for conv_size in self.cnn_layer_w_para:
                in_plus_out = self.embadding_num + self.embadding_num * conv_size[0] * conv_size[1]
                border = np.sqrt(6.0 / in_plus_out)
                init_w = np.random.uniform(-border, border, [conv_size[0], self.embadding_num, 1, conv_size[1]])
                init_b = tf.zeros([conv_size[1]])
                w = tf.Variable(init_w, name="conv_w")
                b = tf.Variavle(init_b, name="conv_b")
                conv = tf.nn.conv2d(self.inputs, w, strides=[1,1,1,1], padding="VALID", name="conv") + b
                activate = tf.tanspose(tf.nn.relu(conv), [0, 3, 2, 1])
                max_pooling = tf.nn.top_k(activate, self.max_k, sorted=False, name="pooling")
                reduce_dim = tf.reshape(max_pooling, [None, np.prod(max_pooling.shape.as_list()[1:])])
                self.cnn_outs.append(reduce_dim)
            # concat
            concat = reduce(lambda x, y: tf.concat([x, y], axis=1), self.cnn_outs)
            self.cnn_concat = concat
            dense_inputs = concat
            for out_num in self.dense_hidden_nums:
                in_num = int(dense_inputs.shape[-1])
                border = np.sqrt(6.0/(out_num + in_num))
                init_w = np.random.uniform(-border, border, [in_num, out_num])
                init_b = np.zeros([out_num])
                w = tf.Variable(init_w, name="dense_w")
                b = tf.Variavle(init_b, name="dense_b")
                dense_inputs = tf.nn.tanh(tf.matmul(dense_inputs, w) + b)            
            # sentence level attention
            pass
            # lstm decoder
            # C can be the features of all context 
            

