# coding: utf-8
import Amadeus
from Amadeus.brain.talk.seq2seq_attention import Seq2SeqAttentionModel
from server import load_conf
import os
import tensorflow as tf
import sys


def load_model():
    save_path = load_conf.load_model_path()
    model_file = tf.train.latest_checkpoint(save_path)
    # print("\n\n==============load!=================")
    # print(model_file)
    sys.stderr.write("loading: " + model_file + "\n")
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    rnn_size = 256
    layer_size = 2
    start = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    end = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 1
    pad = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 2
    uk = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 3
    encoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    decoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    embedding_dim = 256
    grad_clip = 5
    max_step = 20
    beam_width = 3
    with tf.device("/gpu:0"):
        with tf.variable_scope("s2s_model", reuse=None, initializer=initializer):
            S2S = Seq2SeqAttentionModel(rnn_size, layer_size,
                                    encoder_vocab_size, decoder_vocab_size,
                                    embedding_dim,
                                    grad_clip,
                                    start, end, pad, uk, max_step,
                                    beam_width,
                                    interface=False)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=3)
    saver.restore(sess, model_file)
    # print("load Done")
    return S2S, sess
