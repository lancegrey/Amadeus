# coding: utf-8
import Amadeus
from scripts.train_seq2seq_attention import init_main
from Amadeus.brain.talk.seq2seq_attention import Seq2SeqAttentionModel
from server import load_conf
import os
import tensorflow as tf
import sys


def load_model():
    save_path = load_conf.load_model_path()
    print(save_path)
    save_path = "E:/PySpace/Amadeus/model_fenci/"
    model_file = "E:/PySpace/Amadeus/model_fenci/model-90813"
    # print("\n\n==============load!=================")
    print(model_file)
    sys.stderr.write("loading: " + model_file + "\n")
    S2S = init_main(interface=True, beam_width=3)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=3)
    saver.restore(sess, model_file)
    # print("load Done")
    return S2S, sess
