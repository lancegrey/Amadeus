# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class Seq2SeqAttentionModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self, rnn_size, layer_size, encoder_vocab_size,
                 decoder_vocab_size, embedding_dim,
                 grad_clip, start, end, pad, uk, max_step=50, beam_width=3, interface=False):
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
        self.interface = interface  # saver是个坑爹玩意。。。2019.5.4--02:39 debug done!
        # placeholder
        self.enc_input = tf.placeholder(tf.int32, shape=[None, None], name='enc_input')
        self.enc_size = tf.placeholder(tf.int32, shape=[None], name='enc_size')
        self.dec_input = tf.placeholder(tf.int32, shape=[None, None], name='dec_input')
        self.dec_state = tf.placeholder(tf.float32, shape=[4, rnn_size], name='dec_state')
        self.dec_label = tf.placeholder(tf.int32, shape=[None, None], name='dec_label')
        self.dec_size = tf.placeholder(tf.int32, shape=[None], name='dec_size')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="leaning_rate")
        self.dec_batch_size = tf.placeholder(tf.int32, [], name='dec_batch_size')
        self.dec_batch_inputs = tf.ones(self.dec_batch_size, dtype=tf.int32, name="dec_batch_inputs") * self.start

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size), output_keep_prob=self.keep_prob)) for _ in range(layer_size)])
        self.enc_cell_fw = tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size), output_keep_prob=self.keep_prob))
        self.enc_cell_bw = tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size), output_keep_prob=self.keep_prob))
        self.attn_projection = tf.layers.Dense(self.rnn_size,
                                               dtype=tf.float32,
                                               use_bias=False,
                                               name='attention_cell_input_fn')
        # 词向量
        self.encoder_embedding = tf.get_variable(
            "e_emb", [encoder_vocab_size, embedding_dim])
        if False:
            self.decoder_embedding = self.encoder_embedding
        else:
            self.decoder_embedding = tf.get_variable(
                "d_emb", [decoder_vocab_size, embedding_dim])

        # 定义softmax层的变量
        self.dense_before_softmax = tf.layers.Dense(decoder_vocab_size)

        with tf.variable_scope("forward"):
            ret = self.forward(self.enc_input,
                               self.enc_size,
                               self.dec_input,
                               self.dec_label,
                               self.dec_size,
                               self.keep_prob,
                               self.learning_rate)
            prob, beam_outputs, avg_loss, train_op, debug_enc_state, logits = ret

            self.train_prob = prob
            self.beam_outputs = beam_outputs
            self.avg_loss = avg_loss
            self.train_op = train_op
            self.debug_enc_state = debug_enc_state
            self.logits = logits

    def cell_input_fn(self, inputs, attention):
                return self.attn_projection(array_ops.concat([inputs, attention], -1))

    def forward(self, e_input, e_size, d_input, d_label, d_size, keep_prob, learning_rate):
        batch_size = tf.shape(e_input)[0]

        e_emb = tf.nn.embedding_lookup(self.encoder_embedding, e_input)
        d_emb = tf.nn.embedding_lookup(self.decoder_embedding, d_input)

        #e_emb = tf.nn.dropout(e_emb, keep_prob)
        #d_emb = tf.nn.dropout(d_emb, keep_prob)

        # outputs是最后一层每个step的输出
        # states是每一层最后step的输出
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, e_emb, e_size, dtype=tf.float32)
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)
            debug_enc_state = enc_state

        # 构造解码器
        # if not self.interface:
        with tf.variable_scope("decoder"):
            # for train
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.rnn_size, enc_outputs,
                memory_sequence_length=e_size)  # memory_sequence_length 豁免padding



            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=self.rnn_size,
                cell_input_fn=self.cell_input_fn)

            init_state = attention_cell.zero_state(self.dec_batch_size, tf.float32)\
                .clone(cell_state=enc_state)
            train_helper = TrainingHelper(d_emb, self.dec_size, time_major=False)
            train_decoder = BasicDecoder(attention_cell, train_helper, init_state, self.dense_before_softmax)
            dec_outputs, _, _ = dynamic_decode(train_decoder, impute_finished=True)
            dec_outputs = tf.identity(dec_outputs.rnn_output)
            logits = tf.reshape(dec_outputs, [-1, self.decoder_vocab_size])
            prob = tf.nn.softmax(logits)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(d_label, [-1]), logits=logits)

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
            # beamsearch_outs = None
        # else:
        with tf.variable_scope("decoder", reuse=True):
            # for beamsearch
            beamsearch_decoder = self.interface_beamsearch(enc_outputs, enc_state, e_size)
            beamsearch_outs = dynamic_decode(beamsearch_decoder, maximum_iterations=self.max_step)[0].predicted_ids
            #prob = None
            #avg_loss = None
            #train_op = None
        return prob, beamsearch_outs, avg_loss, train_op, debug_enc_state, dec_outputs

    def train(self, sess, enc_input, enc_size, dec_input, dec_label, dec_size, lr, kp):
        feed_dict = {self.enc_input: enc_input,
                     self.enc_size: enc_size,
                     self.dec_input: dec_input,
                     self.dec_label: dec_label,
                     self.dec_size: dec_size,
                     self.learning_rate: lr,
                     self.keep_prob: kp,
                     self.dec_batch_size: len(enc_size)}
        ret = sess.run([self.avg_loss, self.train_op, self.logits], feed_dict=feed_dict)
        return ret

    def predict(self, sess, enc_input, enc_size):
        feed_dict = {self.enc_input: enc_input,
                     self.enc_size: enc_size,
                     self.dec_batch_size: len(enc_size),
                     self.keep_prob: 1.0}
        return sess.run([self.beam_outputs], feed_dict=feed_dict)

    def interface_beamsearch(self, enc_outputs, enc_state, e_size):
        beam_width = self.beam_width
        enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=self.beam_width)
        enc_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), enc_state)
        e_size = tf.contrib.seq2seq.tile_batch(e_size, multiplier=self.beam_width)
        batch_starts = self.dec_batch_inputs
        softmax_call = self.dense_before_softmax
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.embedding_dim, enc_outputs,
                memory_sequence_length=e_size)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=self.embedding_dim,
                cell_input_fn=self.cell_input_fn)
        init_state = attention_cell.zero_state(self.dec_batch_size*self.beam_width, tf.float32)\
                     .clone(cell_state=enc_state)
        beamsearch = BeamSearchDecoder(attention_cell, self.decoder_embedding, batch_starts,
                                       self.end, init_state, beam_width, softmax_call)
        return beamsearch

    def new_session(self, init_op=None):
        sess = tf.Session()
        if init_op is not None:
            sess.run(init_op)
        return sess
