# coding: utf-8
# This is the tf origin model of Amadeus
# seq2seq

import Amadeus
import tensorflow as tf
import numpy as np
from functools import reduce
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):
    def __init__(self, rnn_size, layer_size, encoder_vocab_size,
        decoder_vocab_size, embedding_dim, grad_clip, start, end, pad, mask):
        # init_paras
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.grad_clip = grad_clip
        self.end = end
        self.start = start
        self.pad = pad
        self.mask = mask

        # define inputs
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

        # define embedding layer
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embedding_dim], stddev=0.1),
                name='encoder_embedding')
            decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1),
                name='decoder_embedding')

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(rnn_size, layer_size)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(encoder_embedding, self.input_x)
        # print(input_x_embedded.shape)
        # exit()
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)
        self.encoder_state = encoder_state
        # define interface helper for decoder
        self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
        i_helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, end)
        # define train helper for decoder
        self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
        with tf.device('/cpu:0'):
            target_embeddeds = tf.nn.embedding_lookup(decoder_embedding, self.target_ids)
        t_helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size)
            decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
            i_decoder = BasicDecoder(decoder_cell, i_helper, encoder_state, fc_layer)
            t_decoder = BasicDecoder(decoder_cell, t_helper, encoder_state, fc_layer)
            self.decoder_w = fc_layer.trainable_variables

        i_logits, _, _ = dynamic_decode(i_decoder)
        logits, final_state, final_sequence_lengths = dynamic_decode(t_decoder)

        # train-net
        targets = tf.reshape(self.target_ids, [-1])
        logits_flat = tf.reshape(logits.rnn_output, [-1, decoder_vocab_size])
        self.rnn_out = logits_flat
        self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
        tvars = tf.trainable_variables()
        grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads = grads
        self.tvars = tvars
        # optimizer = tf.train.GradientDescentOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # interface-net
        self.prob = i_logits

        # initial
        self.init_op = tf.global_variables_initializer()

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)

    def train(self, sess, inputx,
              target_ids, decoder_seq_length):
        feed_dict = {self.input_x: inputx, self.target_ids: target_ids,
                     self.decoder_seq_length: decoder_seq_length}
        ret = sess.run([self.cost, self.encoder_state, self.rnn_out, self.global_norm, self.grads, self.decoder_w, self.tvars], feed_dict=feed_dict)
        return ret

    def new_session(self):
        sess = tf.Session()
        sess.run(self.init_op)
        return sess

