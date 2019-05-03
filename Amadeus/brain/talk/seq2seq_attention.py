# coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import TrainingHelper
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.python.util import nest


class Seq2SeqAttentionModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self, rnn_size, layer_size, encoder_vocab_size,
                 decoder_vocab_size, embedding_dim, share_decoder_emb,
                 grad_clip, start, end, pad, uk, max_step=100, beam_width=3):
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.grad_clip = grad_clip
        self.end = end
        self.start = start
        self.pad = pad
        self.uk = uk
        self.max_step = max_step
        self.beam_width = beam_width
        # placeholder
        self.enc_input = tf.placeholder(tf.int32, shape=[None, None], name='enc_input')
        self.enc_size = tf.placeholder(tf.int32, shape=[None], name='enc_size')
        self.dec_input = tf.placeholder(tf.int32, shape=[None, None], name='dec_input')
        self.dec_state = tf.placeholder(tf.float32, shape=[4, rnn_size], name='dec_state')
        self.dec_label = tf.placeholder(tf.int32, shape=[None, None], name='dec_label')
        self.dec_size = tf.placeholder(tf.int32, shape=[None], name='dec_size')
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.dec_batch_size = tf.placeholder(tf.int32, [], name='dec_batch_size')
        self.dec_batch_inputs = tf.ones(self.dec_batch_size, dtype=tf.int32) * self.start
        # 定义编码器和解码器所使用的LSTM结构
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(rnn_size) for _ in range(layer_size)])
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

        # 词向量
        self.encoder_embedding = tf.get_variable(
            "e_emb", [encoder_vocab_size, embedding_dim])
        self.decoder_embedding = tf.get_variable(
            "d_emb", [decoder_vocab_size, embedding_dim])

        # 定义softmax层的变量
        if share_decoder_emb:
            self.softmax_weight = tf.transpose(self.decoder_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight", [embedding_dim, decoder_vocab_size])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [decoder_vocab_size])
        with tf.variable_scope("forward"):
            ret = self.forward(self.enc_input,
                               self.enc_size,
                               self.dec_input,
                               self.dec_label,
                               self.dec_size,
                               self.keep_prob,
                               self.learning_rate)
            prob, beam_outputs, avg_loss, train_op = ret

            self.train_prob = prob
            self.beam_outputs = beam_outputs
            self.avg_loss = avg_loss
            self.train_op = train_op

    def forward(self, e_input, e_size, d_input, d_label, d_size, keep_prob, learning_rate):
        batch_size = tf.shape(e_input)[0]
        # 将输入和输出单词编号转为词向量。
        e_emb = tf.nn.embedding_lookup(self.encoder_embedding, e_input)
        d_emb = tf.nn.embedding_lookup(self.decoder_embedding, d_input)
        # 在词向量上进行dropout。
        e_emb = tf.nn.dropout(e_emb, keep_prob)
        d_emb = tf.nn.dropout(d_emb, keep_prob)

        # outputs是最后一层每个step的输出 [batch_size，step，HIDDEN_SIZE]
        # states是每一层最后step的输出
        with tf.variable_scope("encoder"):
            # 维度都是[batch_size,max_time,HIDDEN_SIZE], 代表两个LSTM在每一步的输出
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, e_emb, e_size, dtype=tf.float32)
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        # 使用dyanmic_rnn构造解码器。
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是 [batch_size, max_time,HIDDEN_SIZE]。
        # initial_state=enc_state
        # for train
        with tf.variable_scope("decoder"):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.embedding_dim, enc_outputs,
                memory_sequence_length=e_size)  # memory_sequence_length 豁免padding
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=self.embedding_dim)

            init_state = attention_cell.zero_state(self.dec_batch_size, tf.float32)\
                .clone(cell_state=enc_state)
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, d_emb, d_size,
                                               initial_state=init_state,
                                               dtype=tf.float32)
            # for beamsearch
            beamsearch_decoder = self.interface_beamsearch(enc_outputs, enc_state, e_size)
            beamsearch_outs = dynamic_decode(beamsearch_decoder)

        output = tf.reshape(dec_outputs, [-1, self.embedding_dim])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        prob = tf.nn.softmax(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(d_label, [-1]), logits=logits)
        # 在计算平均损失时，需要将填充位置的权重设置为0
        label_weights = tf.sequence_mask(d_size, maxlen=tf.shape(d_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        loss = tf.reduce_sum(loss * label_weights)   # 线性计算不影响梯度方向
        avg_loss = loss / tf.reduce_sum(label_weights)  # debug的时候看单个token的loss
        loss = loss / tf.to_float(batch_size)  # 算梯度用batch的平均
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return prob, beamsearch_outs, avg_loss, train_op

    def train(self, sess, enc_input, enc_size, dec_input, dec_label, dec_size, lr, kp):
        feed_dict = {self.enc_input: enc_input,
                     self.enc_size: enc_size,
                     self.dec_input: dec_input,
                     self.dec_label: dec_label,
                     self.dec_size: dec_size,
                     self.learning_rate: lr,
                     self.keep_prob: kp,
                     self.dec_batch_size: len(enc_size)}
        ret = sess.run([self.train_prob, self.avg_loss, self.train_op], feed_dict=feed_dict)
        return ret

    def interface(self, cell):
        # 这里time是主维度
        init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        init_array = init_array.write(0, tf.zeros(self.dec_batch_size, dtype=tf.int32))
        init_loop_var = (cell.zero_state(batch_size=self.dec_batch_size, dtype=tf.float32),
                         init_array, 0)

        def continue_loop_condition(state, dec_ids, step):
            return tf.reduce_all(tf.logical_and(
                tf.not_equal(dec_ids.read(step), self.end),
                tf.less(step, self.max_step-1)))

        def loop_body(state, dec_ids, step):
            dec_input = [dec_ids.read(step)]
            dec_emb = tf.nn.embedding_lookup(self.encoder_embedding, dec_input)
            dec_outputs, next_state = cell.call(state=state, inputs=dec_emb)
            output = tf.reshape(dec_outputs, [-1, self.embedding_dim])
            logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
            next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
            dec_ids = dec_ids.write(step+1, next_id[0])
            return next_state, dec_ids, step+1
        state, dec_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
        return dec_ids.stack()

    def interface_beamsearch(self, enc_outputs, enc_state, e_size):
        beam_width = self.beam_width
        enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=self.beam_width)
        enc_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), enc_state)
        e_size = tf.contrib.seq2seq.tile_batch(e_size, multiplier=self.beam_width)
        batch_starts = self.dec_batch_inputs
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.embedding_dim, enc_outputs,
                memory_sequence_length=e_size)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=self.embedding_dim)
        init_state = attention_cell.zero_state(self.dec_batch_size, tf.float32)\
                .clone(cell_state=enc_state)
        beamsearch = BeamSearchDecoder(attention_cell, self.decoder_embedding, batch_starts,
                                       self.end, init_state, beam_width)
        return beamsearch

    def new_session(self, init_op=None):
        if init_op is None:
            init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        return sess
