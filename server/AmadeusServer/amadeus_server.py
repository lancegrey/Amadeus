# coding: utf-8
import sys
import json
import socket
import Amadeus
from Amadeus.brain import dictionary
from Amadeus.brain.wordseg.base_wordseg import JiebaSeg
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

    def predict(self, inputs):
        data = []
        for line in inputs:
            data.append(self.sentence_to_id(line))
        label = self.S2S.predict(self.sess, data)
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
        words = []
        # 转换成str
        for i in ids:
            word = self.reserve_dic.get(i, None)
            if word is not None:
                words.append(word)
        return "".join(words)


class AmadeusServer(BaseAmadeusServer):
    def __init__(self):
        super(AmadeusServer).__init__()
        self.ip, self.port = load_conf.load_amadeus_ip_port()
        self.recv_buffer_size = load_conf.load_amadeus_recv_buffer_size()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind()

    def _bind(self):
        self.sock.bind((self.ip, self.port))

    def _handle_request(self, connection):
        buf = connection.recv(self.recv_buffer_size)
        try:
            qus = json.loads(buf)
            ans = self.predict(qus)
            ret_buf = json.dumps(ans)
        except Exception as e:
            ret_buf = json.dumps({"error": "1"})
        connection.sendall(ret_buf)

    def service(self):
        self.sock.listen(1)
        while True:
            connection, address = self.sock.accept()
            self._handle_request(connection)
            connection.close()


if __name__ == "__main__":
    server = AmadeusServer()
    server.service()
