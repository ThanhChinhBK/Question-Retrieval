import tensorflow as tf

class ESIMEncoder(object):

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
        with tf.variable_scope("encoded"):
            fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            # encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(
            # lstm_cell_question, question, masks_question, dtype=tf.float32)
            # # (-1, Q, H)
            (encoded_question_f, encoded_question_b), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell,
                bw_lstm_cell,
                question, masks_question,
                dtype=tf.float32
            )
            encoded_question = tf.concat(
                (encoded_question_f, encoded_question_b), -1)  # (-1, P, 2*H)
 
        with tf.variable_scope("encoded", reuse=True):
            (encoded_hypothesis_f, encoded_hypothesis_b), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell, bw_lstm_cell,
                hypothesis, masks_hypothesis,
                dtype=tf.float32)
            encoded_hypothesis = tf.concat(
                (encoded_hypothesis_f, encoded_hypothesis_b), -1)  # (-1, P, 2*H)

        # outputs beyond sequence lengths are masked with 0s
        encoded_question = tf.tanh(
            tf.layers.batch_normalization(encoded_question))
        encoded_hypothesis = tf.tanh(
            tf.layers.batch_normalization(encoded_hypothesis))
        encoded_question = tf.nn.dropout(encoded_question, self.dropout)
        encoded_hypothesis = tf.nn.dropout(encoded_hypothesis, self.dropout)

        # Scoring
        Eph = tf.matmul(encoded_question, encoded_hypothesis, transpose_b=True)
        Eh = tf.nn.softmax(Eph) # (-1, P, P)
        Ep = tf.nn.softmax(tf.transpose(Eph, perm=[0,2,1])) #(-1, P, P)

        # Normalize score matrix, encoder premesis and get alignment
        QuesAlign = tf.matmul(Ep, encoded_hypothesis)
        HypoAlign = tf.matmul(Eh, encoded_question)
        mm_1 = encoded_question * QuesAlign
        mm_2 = encoded_hypothesis * HypoAlign
        sb_1 = encoded_question - QuesAlign
        sb_2 = encoded_hypothesis - HypoAlign
        QuesAlign = tf.concat([encoded_question, QuesAlign, mm_1, sb_1], -1)
        HypoAlign = tf.concat([encoded_hypothesis, HypoAlign, mm_1, sb_1], -1)
        Compressed =  tf.layers.Dense(self.hidden_size*2, activation=tf.nn.relu)
        QuesAlign = Compressed(QuesAlign)
        HypoAlign = Compressed(HypoAlign)

        QuesAlign = tf.nn.dropout(QuesAlign, self.dropout)
        HypoAlign = tf.nn.dropout(HypoAlign, self.dropout)
        return QuesAlign, HypoAlign


class ESIMDecoder(object):

    def __init__(self, hidden_size, Ddim, dropout, initializer=lambda: None):
        self.hidden_size = hidden_size
        self.init_weights = initializer
        self.Ddim = Ddim
        self.dropout = dropout

    def decode(self, encoded_rep, masks):
        encoded_question, encoded_hypothesis = encoded_rep
        masks_question, masks_hypothesis = masks
        masks_hypothesis = tf.reduce_sum(masks_hypothesis, -1)
        masks_question = tf.reduce_sum(masks_question, -1)
        
        with tf.variable_scope("decoded"):
            fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, state_is_tuple=True)
            # encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(
            # lstm_cell_question, question, masks_question, dtype=tf.float32)
            # # (-1, Q, H)
            (decoded_question_f, decoded_question_b), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell,bw_lstm_cell,
                encoded_question, masks_question,
                dtype=tf.float32
            )
            decoded_question = tf.concat(
                (decoded_question_f, decoded_question_b), -1)  # (-1, P, 2*H)
 
        with tf.variable_scope("decoded", reuse=True):
            (decoded_hypothesis_f, decoded_hypothesis_b), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell, bw_lstm_cell,
                encoded_hypothesis, masks_hypothesis,
                dtype=tf.float32)
            decoded_hypothesis = tf.concat(
                (decoded_hypothesis_f, decoded_hypothesis_b), -1)  # (-1, P, 2*H)

        # outputs beyond sequence lengths are masked with 0s
        decoded_question = tf.tanh(
            tf.layers.batch_normalization(decoded_question))
        decoded_hypothesis = tf.tanh(
            tf.layers.batch_normalization(decoded_hypothesis))
        decoded_question = tf.nn.dropout(decoded_question, self.dropout)
        decoded_hypothesis = tf.nn.dropout(decoded_hypothesis, self.dropout)
        max_q = tf.reduce_max(decoded_question, 1)
        mean_q = tf.reduce_mean(decoded_question, 1)
        max_h = tf.reduce_max(decoded_hypothesis, 1)
        mean_h = tf.reduce_mean(decoded_hypothesis, 1)
        final_vec = tf.concat([max_q, mean_q, max_h, mean_h], -1)
        logits = tf.layers.dense(final_vec, 1)
        return logits


class ESIM(object):

    def __init__(self, flags, vocab, char_vocab, word_embedding):
        self.config = flags
        self.Ddim = [int(x) for x in self.config.Ddim.split()]
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
        
        self._add_embedding()
        self.encoder = ESIMEncoder(self.config.hidden_layer, self.dropout)
        self.decoder = ESIMDecoder(self.config.hidden_layer * 2,
                               self.Ddim, self.dropout)
        self._build_model()
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate).minimize(self.loss)


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
        encoded_queries, encoded_hypothesis = self.encoder.encode(
            [self.queries_embedding, self.hypothesis_embedding],
            [self.queries_length, self.hypothesis_length],
            encoder_state_input=None
        )
        logits = self.decoder.decode([encoded_queries, encoded_hypothesis],
                                     [self.queries_length, self.hypothesis_length])
        self.yp = tf.nn.sigmoid(logits)
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y[:, tf.newaxis],
                                                        logits=logits)
            )
            vars = tf.trainable_variables()
            print("Number of parameter is {}".format(len(vars)))
            vars_SemEval = [v for v in vars
                            if 'bias' not in v.name and "embedd" not in v.name ]
           
            print("Number of parameter in SemEval task is {}".format(
                len(vars_SemEval)))
            lossL2_SemEval = tf.add_n([tf.nn.l2_loss(v)
                                       for v in vars_SemEval]) * 1e-4
            self.loss = self.loss + lossL2_SemEval
