# coding: utf-8
import sys
sys.path.append("..")
import Amadeus
import os
import numpy as np
from Amadeus.brain.talk.seq2seq import Seq2SeqModel


def init_model():
    rnn_size = 128
    layer_size = 2
    start = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM
    end = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 1
    pad = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 2
    mask = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 3
    encoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    decoder_vocab_size = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM + 4
    embedding_dim = 128
    grad_clip = 5
    S2S = Seq2SeqModel(rnn_size, layer_size, encoder_vocab_size,
                       decoder_vocab_size, embedding_dim, grad_clip,
                       start, end, pad, mask)
    return S2S


def load_data(inputs, batch_size, start, end, pad, mask):
    max_batch_len = 0
    max_batch_len_d = 0
    batch = {"data": [], "label": []}
    while True:
        for filename in os.listdir(inputs):
            data = np.load(inputs+filename).item()
            # print(data["data"])
            for d, l in zip(data["data"], data["label"]):
                d = [dd[1] if dd[1] >=0 else mask for dd in d]
                l = [start] + [ll[1] if ll[1] > 0 else mask for ll in l] + [end]
                if len(l) > max_batch_len:
                    max_batch_len = len(l)
                if len(d) > max_batch_len_d:
                    max_batch_len_d = len(d)
                batch["data"].append(d)
                batch["label"].append(l)
                if len(batch["data"]) >= batch_size - 1:
                    for j in range(len(batch["data"])):
                        batch["data"][j] = batch["data"][j] + [pad for _ in range(max_batch_len_d - len(batch["data"][j]))]
                        batch["label"][j] = batch["label"][j] + [pad for _ in range(max_batch_len - len(batch["label"][j]))]
                    batch_lens = [len(x) for x in batch["label"]]
                    yield batch["data"], batch["label"], batch_lens
                    # debug
                    # print(batch["data"][0])
                    break
                    batch = {"data": [], "label": []}
                    max_batch_len = 0
                    max_batch_len_d = 0


def main():
    S2S = init_model()
    start, end, pad, mask = S2S.start, S2S.end, S2S.pad, S2S.mask
    sess = S2S.new_session()
    inputs = Amadeus.AMADEUS_TRAIN_DATA_DIR
    batch_size = 1
    data = load_data(inputs, batch_size, start, end, pad, mask)
    for train_data, train_label, batch_lens in data:
        #print("train:")
        loss = S2S.train(sess, train_data, train_label, batch_lens)
        print("=======")
        #print(loss[0])
        #print(loss[3])
        #print(train_label[0])
        #print(loss[1][0][0][0][-10:])
        #print(sum(sum(loss[-1][0])))
        print(loss[4][0][0][0])
        print(loss[6][0][0][0])

if __name__ == "__main__":
    main()
