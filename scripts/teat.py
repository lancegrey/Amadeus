# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from Amadeus.brain.talk.seq2seq_attention import Seq2SeqAttentionModel
import Amadeus


def load_data(inputs, batch_size, start, end, pad, uk):
    batch = {"e_input": [], "d_label": [], "d_input": [], "e_size": [], "d_size": []}
    epoch = 0
    while True:
        for filename in os.listdir(inputs):
            data = np.load(inputs+filename).item()
            for d, l in zip(data["data"], data["label"]):
                ei = [dd[1] if dd[1] >= 0 else uk for dd in d]
                di = [start] + [ll[1] if ll[1] >= 0 else uk for ll in l]
                dl = [ll[1] if ll[1] >= 0 else uk for ll in l] + [end]
                batch["e_size"].append(len(ei))
                batch["d_size"].append(len(di))
                batch["e_input"].append(ei)
                batch["d_input"].append(di)
                batch["d_label"].append(dl)
                if len(batch["e_size"]) >= batch_size - 1:
                    # padding
                    max_e_len = np.max(batch["e_size"])
                    max_d_len = np.max(batch["d_size"])
                    for j in range(len(batch["e_size"])):
                        batch["e_input"][j] = batch["e_input"][j] + [pad for _ in range(max_e_len - len(batch["e_input"][j]))]
                        batch["d_input"][j] = batch["d_input"][j] + [pad for _ in range(max_d_len - len(batch["d_input"][j]))]
                        batch["d_label"][j] = batch["d_label"][j] + [pad for _ in range(max_d_len - len(batch["d_label"][j]))]
                    yield epoch, batch
                    batch = {"e_input": [], "d_label": [],
                             "d_input": [], "e_size": [], "d_size": []}
                    # debug
                    # print(batch["data"][0])
                    break
        epoch += 1


def init_main(interface=False):
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
    max_step = 50
    beam_width = 3
    with tf.device("/gpu:0"):
        with tf.variable_scope("s2s_model", reuse=None, initializer=initializer):
            S2S = Seq2SeqAttentionModel(rnn_size, layer_size,
                                    encoder_vocab_size, decoder_vocab_size,
                                    embedding_dim,
                                    grad_clip,
                                    start, end, pad, uk, max_step,
                                    beam_width,
                                    interface=interface)
    return S2S


def test(S2S, save_path, saver, batch):
    with tf.Graph().as_default():
        S2S = init_main(True)
        print(tf.get_default_graph().collections)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
            model_file = tf.train.latest_checkpoint(save_path)
            print("\n\n=========================================")
            print(model_file)
            saver.restore(sess, model_file)
            print("Done")
            ret = S2S.predict(sess, batch["e_input"], batch["e_size"])
        return ret


def main():
    S2S = init_main(True)
    init = tf.global_variables_initializer()
    start, end, pad, uk = S2S.start, S2S.end, S2S.pad, S2S.uk
    inputs = Amadeus.AMADEUS_TRAIN_DATA_DIR
    batch_size = 1
    data = load_data(inputs, batch_size, start, end, pad, uk)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        lr = 1e-1
        save_path = "E:/PySpace/Amadeus/model/"
        saver = tf.train.Saver(max_to_keep=3)
        model_file = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, model_file)
        # sess.run(init)
        step = 0
        for epoch, batch in data:
            ret = S2S.predict(sess, batch["e_input"], batch["e_size"])
            print(ret[0][0][:6])




if __name__ == "__main__":
    main()
