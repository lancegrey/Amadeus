# coding: utf-8
import sys
import json
import socket
import Amadeus
import numpy as np
from Amadeus.brain import dictionary
from Amadeus.brain.wordseg.base_wordseg import JiebaSeg
from Amadeus.brain.wordseg.base_wordseg import ZSeg
from server import load_conf
from server.AmadeusServer.load_model import load_model


class BaseAmadeusServer(object):
    def __init__(self):
        S2S, sess = load_model()
        self.S2S = S2S
        self.sess = sess
        self.dic = dictionary.Dictionary()
        self.dic.read_from_file(Amadeus.AMADEUS_DICTIONARY)
        self.reserve_dic = {v[0]: k for k, v in self.dic.items()}
        self.wordseg = JiebaSeg()
        #self.wordseg = ZSeg()

    def predict(self, inputs):
        data = []
        lens = []
        for line in inputs:
            data.append(self.sentence_to_id(line))
            lens.append(len(data))
        max_len = np.max(lens)
        pad = self.S2S.pad
        for i in range(len(data)):
            data[i] = data[i] + [pad for _ in range(max_len - lens[i])]
        label = self.S2S.predict(self.sess, data, lens)[0]
        print(label)
        ans = []
        for words in label:
            ans.append(self.id_to_sentence(words))
        return ans

    def sentence_to_id(self, line):
        words = []
        # 转换成向量
        for word in self.wordseg.cut(line.strip()):
            word_id = self.dic.get(word, [-1, -1])[0]
            words.append(word_id)
        return words

    def id_to_sentence(self, ids):
        beam = []
        for loc in range(len(ids[0])):
            words = []
            # 转换成str
            for i in ids:
                word = self.reserve_dic.get(i[loc], None)
                if word is not None:
                    words.append(word)
            beam.append("".join(words))
        return beam


class AmadeusServer(BaseAmadeusServer):
    def __init__(self):
        BaseAmadeusServer.__init__(self)
        self.ip, self.port = load_conf.load_amadeus_ip_port()
        self.recv_buffer_size = load_conf.load_amadeus_recv_buffer_size()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind()

    def _bind(self):
        self.sock.bind((self.ip, self.port))

    def _handle_request(self, connection):
        buf = connection.recv(self.recv_buffer_size)
        #try:
        qus = json.loads(buf.decode())
        ans = self.predict(qus)
        ret_buf = json.dumps(ans)
        #except Exception as e:
        #    ret_buf = json.dumps({"error": "1"})
        #    print(e)
        connection.sendall(ret_buf.encode())

    def service(self):
        self.sock.listen(1)
        print("suc service!")
        while True:
            connection, address = self.sock.accept()
            self._handle_request(connection)
            connection.close()


if __name__ == "__main__":
    server = AmadeusServer()
    server.service()
