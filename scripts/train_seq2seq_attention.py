# coding: utf-8
import os
import numpy as np
import datetime
import tensorflow as tf
from Amadeus.brain.talk.seq2seq_attention import Seq2SeqAttentionModel
import Amadeus
import random


def load_data(inputs, batch_size, max_len, start, end, pad, uk):
    batch = {"e_input": [], "d_label": [], "d_input": [], "e_size": [], "d_size": []}
    epoch = 0
    while True:
        filenames = [i for i in os.listdir(inputs)]
        random.shuffle(filenames)
        for filename in filenames:
            data = np.load(inputs+filename).item()
            for d, l in zip(data["data"], data["label"]):
                ei = [dd[1] if dd[1] >= 0 else uk for dd in d]
                di = [start] + [ll[1] if ll[1] >= 0 else uk for ll in l]
                dl = [ll[1] if ll[1] >= 0 else uk for ll in l] + [end]
                if len(di) < max_len and len(ei) < max_len:
                    batch["e_size"].append(len(ei))
                    batch["d_size"].append(len(di))
                    batch["e_input"].append(ei)
                    batch["d_input"].append(di)
                    batch["d_label"].append(dl)
                if len(batch["e_size"]) >= batch_size:
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
                    # break
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
                                    interface=interface)
    return S2S


def main():
    S2S = init_main()
    init = tf.global_variables_initializer()
    start, end, pad, uk, max_len = S2S.start, S2S.end, S2S.pad, S2S.uk, S2S.max_step
    inputs = Amadeus.AMADEUS_TRAIN_DATA_DIR
    batch_size = 32
    data = load_data(inputs, batch_size, max_len, start, end, pad, uk)
    log = open("E:\\PySpace\\Amadeus\\logs\\log.log", "w")

    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        lr = 1e-2
        save_name = "E:/PySpace/Amadeus/model/model"
        saver = tf.train.Saver(max_to_keep=3)
        sess.run(init)
        step = 0
        last_epoch = -1
        for epoch, batch in data:
            ret = S2S.train(sess, batch["e_input"], batch["e_size"],
                            batch["d_input"], batch["d_label"], batch["d_size"],
                            lr=lr, kp=0.5)
            loss, _ = ret
            if loss < 1.0:
                lr = 1e-4
            elif loss < 2.0:
                lr = 1e-3
            if last_epoch != epoch:
                saver.save(sess, save_name, global_step=step)
                print(batch["d_label"][0])
                ret = S2S.predict(sess, batch["e_input"], batch["e_size"])
                print(ret[0][0][:6])
                print(loss)
                now = datetime.datetime.now()
                log.write("======================\n")
                log.write(str(now) + '\n')
                log.write("loss:" + str(loss) + "\n")
                log.write(str(batch["d_label"][0]) + '\n')
                log.write(str(ret[0][0]) + '\n')
                last_epoch = epoch
            step += 1
            print(step, ":", loss)
    log.close()


if __name__ == "__main__":
    main()
