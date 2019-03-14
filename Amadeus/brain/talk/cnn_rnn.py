# coding: utf-8
# This is the base model of Amadeus
# embadding-conv-pool-rnn

import tensorflow as tf
import numpy as np
from functools import reduce


class TalkModel(object):
    """
    model
    """
    def __init__(self, embadding_num=256, dictionary_size=5000):
        self.embadding_num = embadding_num
        # if sentence len is smaller than k, it's hard to use max-k pooling without padding;so k=1;
        # padding the sentence with zeros-vector before input to ensure that 
        # the length of sentence is biger than all of filter sizes
        self.max_k = 1  
        self.cnn_layer_w_para = [[1, 32], [2, 32], [3, 32], [4, 32], [5, 32]]  # filter size, filter num
        self.dense_hidden_nums = [128]
        self.decoder_max_len = 10
        self.decoder_layer_num = 1
        self.soft_max_samples = 256
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
                w = tf.Variable(init_w, name="conv_w", dtype=tf.float32)
                b = tf.Variable(init_b, name="conv_b", dtype=tf.float32)
                conv = tf.nn.conv2d(self.inputs, w, strides=[1,1,1,1], padding="VALID", name="conv") + b
                activate = tf.transpose(tf.nn.relu(conv), [0, 3, 2, 1])
                max_pooling = tf.nn.top_k(activate, self.max_k, sorted=False, name="pooling").values
                reduce_dim = tf.reshape(max_pooling, [-1, np.prod(max_pooling.shape.as_list()[1:])])
                self.cnn_outs.append(reduce_dim)
            # concat conv features
            concat = reduce(lambda x, y: tf.concat([x, y], axis=1), self.cnn_outs)
            self.cnn_concat = concat
            dense_outputs = concat
            for out_num in self.dense_hidden_nums:
                dense_outputs = self.dense(dense_outputs, out_num, tf.nn.tanh)
            self.dense_outputs = dense_outputs
            # sentence level attention: query=last_sentence;keys=all_sentences_before_and_query;value=key
            pass
        with tf.variable_scope("lstm_decoder"):
            # lstm decoder
            # C can be the features of all context 
            # do not use ht-1 as the input for ht, use label t-1 instead.
            self.dc_input = tf.placeholder(tf.float32, [None, self.decoder_max_len, self.dense_hidden_nums[-1]])
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [self.dense_hidden_nums[-1]]*self.decoder_layer_num]
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)
            state = []
            self.init_h = tf.placeholder(tf.float32, shape=[None, self.dense_hidden_nums[-1]])
            for _ in range(self.decoder_layer_num):
                state.append(tf.nn.rnn_cell.LSTMStateTuple(self.dense_outputs, self.init_h))
            state = tuple(state)
            # for train
            dc_outs, dc_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=self.dc_input, initial_state=state, dtype=tf.float32)
            # tensor can not iter
            # dc_outs_stack = reduce(lambda x, y: tf.concat([x, y]), tf.transpose(dc_outs, [1,0,2]))
            trans_dc_outs = tf.transpose(dc_outs, [1, 0, 2])
            dc_outs_stack = 
            in_num = int(x.shape[-1])
            out_num = dictionary_size
            border = np.sqrt(6.0/(out_num + in_num))
            init_w = np.random.uniform(-border, border, [in_num, out_num])
            init_b = np.zeros([out_num])
            w = tf.Variable(init_w, name="dense_w", dtype=tf.float32)
            b = tf.Variable(init_b, name="dense_b", dtype=tf.float32)
            dc_outs = tf.matmul(dc_outs_stack, w) + b
            self.dc_outs, self.dc_states = dc_outs, dc_states
            # for generate
            self.tdc_out, self.tdc_state = multi_rnn_cell(np.zeros([1, self.dense_hidden_nums[-1]], np.float32), state)
            self.tdc_out = tf.matmul(dc_outs_stack, w) + b
            self.tdc_out = tf.nn.softmax(self.tdc_out)

        with tf.variable_scope("optimize"):
            # sample softmax loss
            #self.dc_label = tf.placeholder(tf.float32, [None, self.dense_hidden_nums[-1]])
            self.dc_label = tf.placeholder(tf.float32, [None, 1])  # sample softmax
            self.sample_loss = tf.nn.sampled_softmax_loss(w.T, b, self.dc_labels, dc_outs_stack, self.softmax_samples, dictionary_size)
            self.optimize_op =tf.train.AdamOptimizer(learning_rate=1e-2, beta1=0.9,beta2=0.999, epsilon=1e-08).minimize(self.sample_loss)
        self.init_op = tf.global_variables_initializer()
        
    def dense(self, x, out_num, activation, init=()):
        if len(init) != 2:
            in_num = int(x.shape[-1])
            border = np.sqrt(6.0/(out_num + in_num))
            init_w = np.random.uniform(-border, border, [in_num, out_num])
            init_b = np.zeros([out_num])
        else:
            init_w, init_b = init
        w = tf.Variable(init_w, name="dense_w", dtype=tf.float32)
        b = tf.Variable(init_b, name="dense_b", dtype=tf.float32)
        return activation(tf.matmul(x, w) + b)            
    
    def new_session(self):
        sess = tf.Session()
        sess.run(self.init_op)
        return sess

    def run_optimize(self, inputs, labels, sess):
        pass
            


if __name__ == "__main__":
    S2S = TalkModel()

