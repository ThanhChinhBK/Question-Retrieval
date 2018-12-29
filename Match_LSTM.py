import tensorflow as tf
#from attention_wrapper import _maybe_mask_score
#from attention_wrapper import *
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops
import numpy as np


rnnact_dict = {
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "relu": tf.nn.relu
}

def cnnsum(inputs, dropout, cnnact=tf.nn.relu,
           cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2, 6: 1/2, 7: 1/2},
           input_dim=300, pad=100, pfx=''):
    si_cnn_res_list = []
    tot_len = 0
    for fl, cd in cdim.items():
        nb_filter = int(input_dim*cd)
        si_conv = tf.layers.conv1d(name=pfx+'conv%d'%(fl), inputs=inputs,
                                   kernel_size=fl, filters=nb_filter)
        si_cnn_one = cnnact(tf.layers.batch_normalization(si_conv))
        
        si_pool_one =  tf.layers.max_pooling1d(si_cnn_one,
                                               pool_size=int(pad-fl+1),
                                               strides=int(pad-fl+1), 
                                               name=pfx+'pool%d'%(fl))

        si_out_one =  tf.contrib.layers.flatten(si_pool_one)

        si_cnn_res_list.append(si_out_one)
    
        tot_len += nb_filter

    si_cnn = tf.nn.dropout(tf.concat(si_cnn_res_list, -1), dropout)

    return si_cnn

def _reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])


class Encoder(object):

    # tf.contrib.layers.xavier_initializer):
    def __init__(self, hidden_size, dropout, initializer=lambda: None):
        self.hidden_size = hidden_size
        self.init_weights = initializer
        self.dropout = dropout

    def encode(self, inputs, masks, encoder_state_input=None):
        """
        :param inputs: vector representations of question and hypothesis (a tuple) 
        :param masks: masking sequences for both question and hypothesis (a tuple)
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of the question and hypothesis.
        """

        question, hypothesis = inputs
        masks_question, masks_hypothesis = masks

        # read hypothesis conditioned upon the question
        with tf.variable_scope("encoded_question"):
            lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            # encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(
            # lstm_cell_question, question, masks_question, dtype=tf.float32)
            # # (-1, Q, H)
            (encoded_question_f,encoded_question_b),((q_rep_f,_),(q_rep_b,_)) = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_question, lstm_cell_question, question, masks_question, dtype=tf.float32)
            encoded_question = tf.concat((encoded_question_f, encoded_question_b), -1) # (-1, P, 2*H)
            q_rep = tf.concat((q_rep_f, q_rep_b), -1)
        with tf.variable_scope("encoded_hypothesis"):
            lstm_cell_hypothesis = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            #encoded_hypothesis, (p_rep, _) = tf.nn.dynamic_rnn(
            #    lstm_cell_hypothesis, hypothesis, masks_hypothesis, dtype=tf.float32)  # (-1, P, H)
            (encoded_hypothesis_f,encoded_hypothesis_b),((p_rep_f,_), (p_rep_b,_))=tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_hypothesis, lstm_cell_hypothesis, hypothesis, masks_hypothesis, dtype=tf.float32)
            encoded_hypothesis = tf.concat((encoded_hypothesis_f, encoded_hypothesis_b), -1) # (-1, P, 2*H)
            p_rep = tf.concat((p_rep_f, p_rep_b), -1)

        # outputs beyond sequence lengths are masked with 0s
        encoded_question = tf.tanh(
            tf.layers.batch_normalization(encoded_question))
        encoded_hypothesis = tf.tanh(
            tf.layers.batch_normalization(encoded_hypothesis))
        q_rep = tf.tanh(tf.layers.batch_normalization(q_rep))
        p_rep = tf.tanh(tf.layers.batch_normalization(p_rep))
        encoded_question = tf.nn.dropout(encoded_question, self.dropout)
        encoded_hypothesis = tf.nn.dropout(encoded_hypothesis, self.dropout)
        q_rep = tf.nn.dropout(q_rep, self.dropout)
        p_rep = tf.nn.dropout(p_rep, self.dropout)
        return encoded_question, encoded_hypothesis, q_rep, p_rep


class Decoder(object):

    def __init__(self, hidden_size, Ddim, dropout, initializer=lambda: None):
        self.hidden_size = hidden_size
        self.init_weights = initializer
        self.Ddim = Ddim
        self.dropout = dropout
        
    def run_lstm(self, encoded_rep, q_rep, masks):
        encoded_question, encoded_hypothesis = encoded_rep
        masks_question, masks_hypothesis = masks

        q_rep = tf.expand_dims(q_rep, 1)  # (batch_size, 1, D)
        encoded_hypothesis_shape = tf.shape(encoded_hypothesis)[1]
        q_rep = tf.tile(q_rep, [1, encoded_hypothesis_shape, 1])

        mixed_question_hypothesis_rep = tf.concat(
            [encoded_hypothesis, q_rep], axis=-1)

        with tf.variable_scope("lstm_"):
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            reverse_mixed_question_hypothesis_rep = _reverse(
                mixed_question_hypothesis_rep, masks_hypothesis, 1, 0)

            output_attender_fw, _ = tf.nn.dynamic_rnn(
                cell, mixed_question_hypothesis_rep, dtype=tf.float32, scope="rnn")
            output_attender_bw, _ = tf.nn.dynamic_rnn(
                cell, reverse_mixed_question_hypothesis_rep, dtype=tf.float32, scope="rnn")

            output_attender_bw = _reverse(
                output_attender_bw, masks_hypothesis, 1, 0)

        output_attender = tf.concat(
            [output_attender_fw, output_attender_bw], axis=-1)  # (-1, P, 2*H)

        return output_attender

    def run_match_lstm(self, encoded_rep, masks, return_sequence=False):
        encoded_question, encoded_hypothesis = encoded_rep
        masks_question, masks_hypothesis = masks

        match_lstm_cell_attention_fn = lambda curr_input, state: tf.concat(
            [curr_input, state], axis=-1)
        
        query_depth = encoded_question.get_shape()[-1]

        # output attention is false because we want to output the cell output
        # and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_match_lstm = BahdanauAttention(
                query_depth, encoded_question, memory_sequence_length=masks_question)
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size*2, state_is_tuple=True)
            lstm_attender = AttentionWrapper(
                cell, attention_mechanism_match_lstm,
                output_attention=False,
                cell_input_fn=match_lstm_cell_attention_fn)

            # we don't mask the hypothesis because masking the memories will be
            # handled by the pointerNet
            reverse_encoded_hypothesis = _reverse(
                encoded_hypothesis, masks_hypothesis, 1, 0)

            output_attender_fw, state_attender_fw = tf.nn.dynamic_rnn(
                lstm_attender, encoded_hypothesis, dtype=tf.float32, scope="rnn")
            output_attender_bw, state_attender_bw = tf.nn.dynamic_rnn(
                lstm_attender, reverse_encoded_hypothesis, dtype=tf.float32, scope="rnn")
            output_attender_bw = _reverse(
                output_attender_bw, masks_hypothesis, 1, 0)
        if return_sequence:
            output_attender = tf.concat(
                [output_attender_fw, output_attender_bw], -1)
        else:
            output_attender = tf.concat(
                [state_attender_fw[0].h, state_attender_bw[0].h], axis=-1)  # (-1, 2*H)
        output_attender = tf.tanh(
            tf.layers.batch_normalization(output_attender))
        output_attender = tf.nn.dropout(output_attender, self.dropout)
        return output_attender

    def run_projection(self, state_attender):
        input_projection = state_attender
        for n, ddim in enumerate(self.Ddim):
            _, curr_dim = input_projection.get_shape()
            w = tf.get_variable(dtype=tf.float32,
                                shape=[curr_dim, self.hidden_size * ddim],
                                name='w_{}'.format(n))
            b = tf.get_variable(dtype=tf.float32,
                                shape=[self.hidden_size * ddim],
                                name='b_{}'.format(n))
            input_projection = tf.matmul(input_projection, w) + b

        _, curr_dim = input_projection.get_shape()
        w_fc = tf.get_variable(
            shape=[curr_dim, 1], name='w_fc', dtype=tf.float32)
        b_fc = tf.get_variable(shape=[1], name='b_fc', dtype=tf.float32)
        output_projection = tf.layers.dense(input_projection,
                                            1,
                                            name="projection_final")
        return output_projection

    def decode(self, encoded_rep, q_rep, masks, labels):
        """
        takes in encoded_rep
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        :param encoded_rep: 
        :param masks
        :param labels
        :return: logits: for each word in hypothesis the probability that it is the start word and end word.
        """

        output_attender = self.run_match_lstm(encoded_rep, masks, True)
        #output_cnn = cnnsum(output_attender, self.dropout)
        output_maxpool = tf.reduce_max(output_attender, 2)
        logits = self.run_projection(output_maxpool)

        return logits


class MatchLSTM(object):

    def __init__(self, flags, vocab, word_embedding):
        self.config = flags
        self.Ddim = [int(x) for x in self.config.Ddim.split()]
        self.vocab = vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
        self.encoder = Encoder(self.config.hidden_layer, self.dropout)
        self.decoder = Decoder(self.config.hidden_layer,
                               self.Ddim, self.dropout)
        self._add_embedding()
        self._build_model()

    def _add_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.queries = tf.placeholder(
                tf.int32, [None, self.config.pad], "queries")
            self.queries_length = tf.placeholder(
                tf.int32, [None], "queries_length")
            self.hypothesis = tf.placeholder(
                tf.int32, [None, self.config.pad], "hypothesis")
            self.hypothesis_length = tf.placeholder(
                tf.int32, [None], "hypothesis_length")
            self.labels = tf.placeholder(
                tf.float32, [None], "labels")
            self.dropout = tf.placeholder(
                tf.float32, [], "dropout")

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
        logits = self.decoder.decode([encoded_queries, encoded_hypothesis],
                                     q_rep,
                                     [self.queries_length, self.hypothesis_length],
                                     self.labels)
        self.pred_labels = tf.nn.sigmoid(logits)
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
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels[:, tf.newaxis],
                                                        logits=logits)
            )
            vars = tf.trainable_variables()
            print("Number of parameter is {}".format(len(vars)))
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name and "embedd" not in v.name]) * 1e-4
            self.loss = self.loss + lossL2
