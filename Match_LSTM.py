import tensorflow as tf
from layers import *
import numpy as np
from match_layer import *


rnnact_dict = {
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "relu": tf.nn.relu
}


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
        masks_question = tf.reduce_sum(masks_question, -1)
        masks_hypothesis = tf.reduce_sum(masks_hypothesis, -1)

        # read hypothesis conditioned upon the question
        with tf.variable_scope("encoded_question"):
            bw_lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            fw_lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            # encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(
            # lstm_cell_question, question, masks_question, dtype=tf.float32)
            # # (-1, Q, H)
            (encoded_question_f, encoded_question_b), ((q_rep_f, _), (q_rep_b, _)) = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell_question, bw_lstm_cell_question, question, masks_question, dtype=tf.float32)
            encoded_question = tf.concat(
                (encoded_question_f, encoded_question_b), -1)  # (-1, P, 2*H)
            q_rep = tf.concat((q_rep_f, q_rep_b), -1)
        with tf.variable_scope("encoded_hypothesis"):
            fw_lstm_cell_hypothesis = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            bw_lstm_cell_hypothesis = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            # encoded_hypothesis, (p_rep, _) = tf.nn.dynamic_rnn(
            # lstm_cell_hypothesis, hypothesis, masks_hypothesis,
            # dtype=tf.float32)  # (-1, P, H)
            (encoded_hypothesis_f, encoded_hypothesis_b), ((p_rep_f, _), (p_rep_b, _)) = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell_hypothesis, bw_lstm_cell_hypothesis, hypothesis, masks_hypothesis, dtype=tf.float32)
            encoded_hypothesis = tf.concat(
                (encoded_hypothesis_f, encoded_hypothesis_b), -1)  # (-1, P, 2*H)
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

    def run_match_lstm(self, encoded_rep, masks, return_sequence=False):
        encoded_question, encoded_hypothesis = encoded_rep
        masks_question, masks_hypothesis = masks
        masks_hypothesis_sum = tf.reduce_sum(masks_hypothesis, -1)
        #masks_question = tf.reduce_sum(masks_question, -1)
        match_lstm_cell_attention_fn = lambda curr_input, state: tf.concat(
            [curr_input, state], axis=-1)

        query_depth = encoded_question.get_shape().as_list()[-1]

        # output attention is false because we want to output the cell output
        # and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_fw = SeqMatchSeqAttention(
                query_depth, encoded_question, masks_question)
            mLSTM_fw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
                                               attention_mechanism_fw)
            attention_mechanism_bw = SeqMatchSeqAttention(
                query_depth, encoded_question, masks_question)
            mLSTM_bw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
                                               attention_mechanism_bw)
            (output_attender_fw, output_attender_bw), (state_attender_fw, state_attender_bw) = tf.nn.bidirectional_dynamic_rnn(
                mLSTM_fw_cell,
                mLSTM_bw_cell,
                encoded_hypothesis,
                sequence_length=masks_hypothesis_sum,
                dtype=tf.float32
            )

            # matchlstm_fw_cell = matchLSTMcell(query_depth, self.hidden_size, encoded_question,
            #                                   masks_question)
            # matchlstm_bw_cell = matchLSTMcell(query_depth, self.hidden_size, encoded_question,
            #                                   masks_question)
            # (output_attender_fw, output_attender_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #     matchlstm_fw_cell,
            #     matchlstm_bw_cell,
            #     encoded_hypothesis,
            #     sequence_length=masks_hypothesis,
            #     dtype=tf.float32
            # )
            # cell_fw = MatchLSTMAttnCell(self.hidden_size, encoded_question, masks_question)
            # cell_bw = MatchLSTMAttnCell(self.hidden_size, encoded_question, masks_question)
            # (output_attender_fw, output_attender_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
            #                                                                               inputs=encoded_hypothesis,
            #                                                                               sequence_length=masks_hypothesis,
            #                                                                               dtype=tf.float32)
        if return_sequence:
            output_attender = tf.concat(
                [output_attender_fw, output_attender_bw], -1)
        else:
            output_attender = tf.concat(
                [state_attender_fw[0].h, state_attender_bw[0].h], axis=-1)  # (-1, 2*H)
        output_attender = tf.tanh(
            tf.layers.batch_normalization(output_attender))
        # self_attention_inputs = output_attender
        # self_attention_depth = output_attender.get_shape().as_list()[-1]
        # print("self attention depth", self_attention_depth)
        # with tf.variable_scope("self_match_lstm_attender"):
        #     self_attention_mechanism_fw = SeqMatchSeqAttention(
        #         self_attention_depth, self_attention_inputs, masks_hypothesis)
        #     self_mLSTM_fw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
        #                                             self_attention_mechanism_fw)
        #     self_attention_mechanism_bw = SeqMatchSeqAttention(
        #         self_attention_depth, self_attention_inputs, masks_hypothesis)
        #     self_mLSTM_bw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
        #                                             self_attention_mechanism_bw)
        #     (output_attender_fw, output_attender_bw), (state_attender_fw, state_attender_bw) = tf.nn.bidirectional_dynamic_rnn(
        #         self_mLSTM_fw_cell,
        #         self_mLSTM_bw_cell,
        #         self_attention_inputs,
        #         sequence_length=masks_hypothesis_sum,
        #         dtype=tf.float32
        #     )
        # output_attender = tf.concat(
        #         [output_attender_fw, output_attender_bw], -1)
        # output_attender = tf.nn.dropout(output_attender, self.dropout)
        return output_attender

    def run_answer_ptr(self, output_attender, masks, scope="ans_ptr"):
        with tf.variable_scope(scope):
            batch_size = tf.shape(output_attender)[0]
            masks_question, masks_passage = masks
            #masks_passage = tf.reduce_sum(masks_passage, -1)
            fake_inputs = tf.zeros((batch_size, 2, 1))
            fake_sequence_length = tf.ones(
                (batch_size,), dtype=tf.int32) * 2  # 2 for start and end

            answer_ptr_cell_input_fn = lambda curr_input, context: context
            query_depth_answer_ptr = output_attender.get_shape()[-1]

            with tf.variable_scope("answer_ptr_attender"):
                with tf.variable_scope("fw"):
                    answer_ptr_cell_fw = PointerNetLSTMCell(
                        self.hidden_size, output_attender, masks_passage)
                    outputs_fw, _ = custom_dynamic_rnn(answer_ptr_cell_fw,
                                                      fake_inputs,
                                                      fake_sequence_length,
                    )
                with tf.variable_scope("bw"):
                    answer_ptr_cell_bw = PointerNetLSTMCell(
                        self.hidden_size, output_attender, masks_passage)
                    outputs_bw, _ = custom_dynamic_rnn(answer_ptr_cell_bw,
                                                      fake_inputs,
                                                      fake_sequence_length,
                    )
            start_prob = (outputs_fw[0:, 0, 0:] + outputs_bw[0:, 1, 0:]) / 2
            end_prob = (outputs_fw[0:, 1, 0:] + outputs_bw[0:, 0, 0:]) / 2

        return start_prob, end_prob

    def run_projection(self, state_attender, output_dim, scope="projection"):
        with tf.variable_scope(scope):
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
            output_projection = tf.layers.dense(input_projection,
                                                output_dim,
                                                name="projection_final")
            output_projection = tf.nn.dropout(output_projection, self.dropout)
        return output_projection

    def decode(self, encoded_rep, q_rep, masks):
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
        output_maxpool = tf.reduce_max(output_attender, 1)
        output_meanpool = tf.reduce_mean(output_attender, 1)
        outputs = tf.concat([output_maxpool, output_meanpool], -1)
        ourputs = tf.nn.dropout(outputs, self.dropout)
        logits = self.run_projection(outputs, 1, "SemEval_projection")
        logits_SNLI = self.run_projection(outputs, 3, "SNLI_projection")
        logits_SQUAD = self.run_answer_ptr(
            output_attender, masks, "SQUAD_ans_ptr")
        return logits, logits_SNLI, logits_SQUAD



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
        self.decoder = Decoder(self.config.hidden_layer ,
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
                tf.int32, [None, self.config.pad_question], "queries")
            self.queries_char = tf.placeholder(
                tf.int32, [None, self.config.pad_question, self.config.char_pad], "queries")
            # self.queries_length = tf.placeholder(
            #    tf.int32, [None], "queries_length")
            self.hypothesis = tf.placeholder(
                tf.int32, [None, self.config.pad_sentence], "hypothesis")
            self.hypothesis_char = tf.placeholder(
                tf.int32, [None, self.config.pad_sentence, self.config.char_pad], "queries")
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
                cq_emb, [-1, self.config.pad_question, 2 * self.config.char_embedding_dim])
            ch_emb = tf.reshape(
                ch_emb, [-1, self.config.pad_sentence, 2 * self.config.char_embedding_dim])
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
