# coding: utf-8
# word to vector by tensorflow

import tensorflow as tf
import numpy as np
import Amadeus
from Amadeus.brain import dictionary
from Amadeus.data import yuliao_process


class W2V(object):
    def __init__(self, wordsnum=Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM, n=4):
        # configure
        self.wordsnum = wordsnum+1
        self.hiddennums = [512, 128]
        self.n = n

        # 1 to n, represent
        self.inputs = tf.placeholder(tf.float32, [None, self.wordsnum])
        self.ws = []
        self.bs = []
        with tf.variable_scope("represent"):
            layer_pointer = self.inputs
            last_num = self.wordsnum
            for num in self.hiddennums:
                std = 2.0 / ((1 + 0.0**2) * num)
                init_w = tf.truncated_normal([last_num, num], mean=0.0, stddev=std)
                init_b = np.zeros([1, num])
                self.ws.append(tf.Variable(init_w, dtype=tf.float32))
                self.bs.append(tf.Variable(init_b, dtype=tf.float32))
                layer_pointer = tf.add(tf.matmul(layer_pointer, self.ws[-1]), self.bs[-1])
                layer_pointer = tf.nn.relu(layer_pointer)
                last_num = num
        self.represent = layer_pointer

        with tf.variable_scope("outputs"):
            self.outputs = []
            for _ in range(self.n):
                std = 2.0 / ((1 + 0.0**2) * self.hiddennums[-1])
                init_w = tf.truncated_normal([self.hiddennums[-1], self.wordsnum], mean=0.0, stddev=std)
                init_b = np.zeros([1, self.wordsnum])
                self.outputs.append(tf.add(tf.matmul(layer_pointer,
                                                     tf.Variable(init_w, dtype=tf.float32)),
                                           tf.Variable(init_b, dtype=tf.float32)))
                self.outputs[-1] = tf.nn.softmax(self.outputs[-1])

        with tf.variable_scope("train"):
            self.labels = tf.placeholder(tf.float32, [self.n, None, self.wordsnum])
            loss = None
            for k in range(self.n):
                label, out = self.labels[k], self.outputs[k]
                if loss is None:
                    loss = tf.reduce_mean(-self.labels * tf.log(out + 1e-10))
                else:
                    loss += tf.reduce_mean(-self.labels * tf.log(out + 1e-10))
            self.loss = loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
            self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, data, labels):
        feed_dict = {self.inputs: data, self.labels: labels}
        return self.sess.run([self.train_op, self.loss, self.represent, self.outputs[0]], feed_dict=feed_dict)

    def get_represent(self, data, dict=None):
        new_data = []
        if dict is not None:
            for word in data:
                loc = dict.get(word, [-1])[0]
                pa = np.zeros(addn)
                if loc >= 0:
                    pa[loc] = 1
                new_data.append(pa)
            data = new_data
        return self.sess.run(self.represent, feed_dict={self.inputs: data})

    def cos(self, x, y, dict=None):
        if dict is not None:
            x = self.get_represent([x], dict)[0]
            y = self.get_represent([y], dict)[0]
        up = np.dot(x, y)
        down = sum(x**2)**0.5 * sum(y**2)**0.5
        return up/down


if __name__ == "__main__":
    import os
    DIC = dictionary.Dictionary()
    DIC.read_from_file(Amadeus.AMADEUS_DICTIONARY)
    print("字典大小:", len(DIC))
    print(sorted(list(DIC.items()), key=lambda x: x[1][0]))

    inputs = Amadeus.AMADEUS_YULIAO
    ws = Amadeus.brain.wordseg.base_wordseg.JiebaSeg()
    n = 4
    batch_size = 64
    w2v = W2V(n=n)
    data = []
    labels = [[] for _ in range(n)]
    addn = Amadeus.AMADEUS_DEFAULT_DICTIONARY_NUM+1
    for filename in os.listdir(inputs):
        try:
            with open(inputs + os.sep + filename, encoding="utf8") as f:
                for line in Amadeus.data.yuliao_process.preprocess_conv(f):
                    for l in line[0] + line[1]:
                        words = []
                        # 转换成向量
                        for word in ws.cut(l.strip()):
                            loc = DIC.get(word, [-1])[0]
                            pa = np.zeros(addn)
                            if loc >= 0:
                                pa[loc] = 1
                            words.append(pa)
                        for i in range(len(words)):
                            data.append(words[i])
                            for j in range(n//2):
                                if i - j < 0:
                                    labels[j].append(np.zeros(addn))
                                else:
                                    labels[j].append(words[i-j])
                                if i + j >= len(words):
                                    labels[n-j-1].append(np.zeros(addn))
                                else:
                                    labels[n-j-1].append(words[i+j])
                    if len(data) >= batch_size:
                        debugs = w2v.train(np.array(data), np.array(labels))
                        print("loss: ", debugs[1])
                        # print(debugs[2])
                        print("sum: ", sum(sum(debugs[2])))
                        print("cos: ", w2v.cos("喜欢", "喜爱", DIC))
                        data = []
                        labels = [[] for _ in range(n)]
        except Exception as e:
            raise e
            print(e)
            continue
        print(filename + " done.")

    print(data)
    print(labels)
    print(np.array(data).shape)
    print(np.array(labels).shape)

    


