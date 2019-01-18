import tensorflow as tf
from layers import *
import numpy as np


rnnact_dict = {
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "relu": tf.nn.relu
}


class MatchLSTM(object):

    def __init__(self, flags, vocab, char_vocab, word_embedding):
        self.config = flags
        self.Ddim = [int(x) for x in self.config.Ddim.split()]
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
        
        self._add_embedding()
        self.encoder = Encoder(self.config.hidden_layer, self.dropout)
        self.decoder = Decoder(self.config.hidden_layer * 2,
                               self.Ddim, self.dropout)
        self._build_model()
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss)
        self.train_op_SNLI = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss_SNLI)
        self.train_op_SQUAD = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss_SQUAD)

    def _add_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.queries = tf.placeholder(
                tf.int32, [None, self.config.pad], "queries")
            self.queries_char = tf.placeholder(
                tf.int32, [None, self.config.pad, self.config.char_pad], "queries")
            # self.queries_length = tf.placeholder(
            #    tf.int32, [None], "queries_length")
            self.hypothesis = tf.placeholder(
                tf.int32, [None, self.config.pad], "hypothesis")
            self.hypothesis_char = tf.placeholder(
                tf.int32, [None, self.config.pad, self.config.char_pad], "queries")
            # self.hypothesis_length = tf.placeholder(
            #    tf.int32, [None], "hypothesis_length")
            self.y = tf.placeholder(
                tf.float32, [None], "labels")
            self.y_SNLI = tf.placeholder(
                tf.float32, [None, 3], "labels_SNLI")
            self.y_SQUAD = tf.placeholder(
                tf.int32, [None, 2], "labels_SNLI")
            self.dropout = tf.placeholder(
                tf.float32, [], "dropout")
            self.queries_length = tf.cast(
                tf.cast(self.queries, tf.bool), tf.int32)
            self.hypothesis_length = tf.cast(
                tf.cast(self.hypothesis, tf.bool), tf.int32)
            self.queries_char_length = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.queries_char, tf.bool), tf.int32), -1), [-1])
            self.hypothesis_char_length = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.hypothesis_char, tf.bool), tf.int32), -1), [-1])

    def _add_embedding(self):
        with tf.variable_scope("char_embeddings") as scope:
            char_embeddings = tf.get_variable("word_embedding",
                                              shape=[self.char_vocab.size(
                                              ), self.config.char_embedding_dim],
                                              dtype=tf.float32)
            cq_embedding = tf.reshape(
                tf.nn.embedding_lookup(char_embeddings, self.queries_char),
                [-1, self.config.char_pad, self.config.char_embedding_dim])
            ch_embedding = tf.reshape(
                tf.nn.embedding_lookup(
                    char_embeddings, self.hypothesis_char),
                [-1, self.config.char_pad, self.config.char_embedding_dim])
            cell_fw = tf.contrib.rnn.GRUCell(self.config.char_embedding_dim)
            cell_bw = tf.contrib.rnn.GRUCell(
                self.config.char_embedding_dim)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, ch_embedding, self.hypothesis_char_length, dtype=tf.float32)
            ch_emb = tf.concat([state_fw, state_bw], axis=1)
            scope.reuse_variables()
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, cq_embedding, self.queries_char_length, dtype=tf.float32)
            cq_emb = tf.concat([state_fw, state_bw], axis=1)
            cq_emb = tf.reshape(
                cq_emb, [-1, self.config.pad, 2 * self.config.char_embedding_dim])
            ch_emb = tf.reshape(
                ch_emb, [-1, self.config.pad, 2 * self.config.char_embedding_dim])
        with tf.variable_scope("word_embeddings"):
            init_emb = tf.constant(self.vocab.embmatrix(
                self.word_embedding), dtype=tf.float32)
            embeddings = tf.get_variable("word_embedding",
                                         initializer=init_emb,
                                         dtype=tf.float32)
            
            self.queries_embedding = tf.nn.embedding_lookup(
                embeddings, self.queries)
            self.hypothesis_embedding = tf.nn.embedding_lookup(
                embeddings, self.hypothesis)
            
        self.queries_embedding = tf.concat([self.queries_embedding, cq_emb], axis=2)
        self.hypothesis_embedding = tf.concat([self.hypothesis_embedding, ch_emb], axis=2)
        self.config.hidden_layer += self.config.char_embedding_dim * 2
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
        logits, logits_SNLI, logits_SQUAD = self.decoder.decode([encoded_queries, encoded_hypothesis],
                                                                q_rep,
                                                                [self.queries_length, self.hypothesis_length])
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
            vars_SemEval = [v for v in vars
                            if 'bias' not in v.name and "embedd" not in v.name and "SNLI" not in v.name and "SQUAD" not in v.name]
            vars_SQUAD = [v for v in vars
                          if 'bias' not in v.name and "embedd" not in v.name and "SNLI" not in v.name and "SemEval" not in v.name]
            print("Number of parameter in SemEval task is {}".format(
                len(vars_SemEval)))
            print("Number of parameter in SQUAD task is {}".format(len(vars_SQUAD)))
            lossL2_SemEval = tf.add_n([tf.nn.l2_loss(v)
                                       for v in vars_SemEval]) * 1e-4
            lossL2_SQUAD = tf.add_n([tf.nn.l2_loss(v)
                                     for v in vars_SQUAD]) * 1e-4
            self.loss = self.loss + lossL2_SemEval
            self.loss_SNLI = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_SNLI,
                logits=logits_SNLI
            ))
            loss_SQUAD_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_SQUAD[:, 0],
                logits=logits_SQUAD[0]
            )
            loss_SQUAD_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_SQUAD[:, 1],
                logits=logits_SQUAD[1]
            )
            self.loss_SQUAD = tf.reduce_mean(loss_SQUAD_1 + loss_SQUAD_2)
