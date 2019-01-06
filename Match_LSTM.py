import tensorflow as tf
from layers import *
import numpy as np


rnnact_dict = {
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "relu": tf.nn.relu
}


class MatchLSTM(object):

    def __init__(self, flags, vocab, word_embedding):
        self.config = flags
        self.Ddim = [int(x) for x in self.config.Ddim.split()]
        self.vocab = vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
        self.encoder = Encoder(self.config.hidden_layer, self.dropout)
        self.decoder = Decoder(self.config.hidden_layer*2,
                               self.Ddim, self.dropout)
        self._add_embedding()
        self._build_model()
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss)
        self.train_op_SNLI = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss_SNLI)

    def _add_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.queries = tf.placeholder(
                tf.int32, [None, self.config.pad], "queries")
            #self.queries_length = tf.placeholder(
            #    tf.int32, [None], "queries_length")
            self.hypothesis = tf.placeholder(
                tf.int32, [None, self.config.pad], "hypothesis")
            #self.hypothesis_length = tf.placeholder(
            #    tf.int32, [None], "hypothesis_length")
            self.y = tf.placeholder(
                tf.float32, [None], "labels")
            self.y_SNLI = tf.placeholder(
                tf.float32, [None, 3], "labels_SNLI")
            self.dropout = tf.placeholder(
                tf.float32, [], "dropout")
            self.queries_length = tf.cast(tf.cast(self.queries, tf.bool), tf.int32)
            self.hypothesis_length = tf.cast(tf.cast(self.hypothesis, tf.bool), tf.int32)

    def _add_embedding(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope("embeddings"):
                init_emb = tf.constant(self.vocab.embmatrix(
                    self.word_embedding), dtype=tf.float32)
                embeddings = tf.get_variable("word_embedding",
                                             initializer=init_emb,
                                             dtype=tf.float32)
                self.queries_embedding = tf.nn.embedding_lookup(
                    embeddings, self.queries)
                self.hypothesis_embedding = tf.nn.embedding_lookup(
                    embeddings, self.hypothesis)
            if self.config.dropout < 1:
                self.queries_embedding = tf.nn.dropout(
                    self.queries_embedding, self.dropout)
                self.hypothesis_embedding = tf.nn.dropout(
                    self.hypothesis_embedding, self.dropout)

    def _build_model(self):
        encoded_queries, encoded_hypothesis, q_rep, h_rep = self.encoder.encode(
            [self.queries_embedding, self.hypothesis_embedding],
            [self.queries_length, self.hypothesis_length],
            encoder_state_input=None
        )
        logits, logits_SNLI = self.decoder.decode([encoded_queries, encoded_hypothesis],
                                     q_rep,
                                     [self.queries_length, self.hypothesis_length])
        print(logits_SNLI.get_shape())
        self.yp = tf.nn.sigmoid(logits)
        self.yp_SNLI = tf.argmax(logits_SNLI, -1)
        with tf.variable_scope("loss"):
            # loss
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=logits) \
            # if self.num_class != 1 else
            # tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
            # logits=logits)
            # cross_entropy = tf.nn.softmax_cr# oss_entropy_with_logits(
            #     labels=self.labels, logits=logits)
            # self.loss = tf.reduce_mean(cross_entropy)
            self.loss = tf.reduce_mean(
                # tf.log(1. + tf.exp(-(self.labels * tf.reshape(self.pred_labels,(-1,)) -
                #                     (1 - self.labels) * tf.reshape(self.pred_
                #)
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y[:, tf.newaxis],
                                                        logits=logits)
            )
            vars = tf.trainable_variables()
            print("Number of parameter is {}".format(len(vars)))
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name and "embedd" not in v.name]) * 1e-4
            self.loss = self.loss + lossL2
            self.loss_SNLI = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_SNLI,
                logits = logits_SNLI
            ))
