# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from Amadeus.brain.talk.seq2seq_attention import Seq2SeqAttentionModel
import Amadeus


def load_data(inputs, batch_size, start, end, pad, uk):
    batch = {"e_input": [], "d_label": [], "d_input": [], "e_size": [], "d_size": []}
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
                    yield batch
                    batch = {"e_input": [], "d_label": [],
                             "d_input": [], "e_size": [], "d_size": []}
                    # debug
                    # print(batch["data"][0])
                    break


def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    rnn_size = 128
    layer_size = 2
    start = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    end = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 1
    pad = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 2
    uk = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 3
    encoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    decoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    embedding_dim = 128
    grad_clip = 5
    with tf.variable_scope("s2s_model", reuse=None, initializer=initializer):
        S2S = Seq2SeqAttentionModel(rnn_size, layer_size,
                                    encoder_vocab_size, decoder_vocab_size,
                                    embedding_dim, False,
                                    grad_clip,
                                    start, end, pad, uk)


    # saver = tf.train.Saver()
    # saver.save(session, CHECKPOINT_PATH, global_step=step)
    inputs = Amadeus.AMADEUS_TRAIN_DATA_DIR
    batch_size = 64
    data = load_data(inputs, batch_size, start, end, pad, uk)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch in data:
            ret = S2S.train(sess, batch["e_input"], batch["e_size"],
                            batch["d_input"], batch["d_label"], batch["d_size"],
                            lr=1e-2, kp=0.5)
            train_prob, loss, _ = ret
            print(loss)
            print(batch["d_label"][0])
            print(np.argmax(train_prob, axis=1)[:len(batch["d_label"][0])])


if __name__ == "__main__":
    main()
