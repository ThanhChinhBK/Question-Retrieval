import tensorflow as tf


class MatchLSTM(object):

    def __init__(self, flags, vocab, word_embedding):
        self.config = flags
        self.vocab = vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
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
                init_emb = tf.constant(self.vocab.embmatrix(self.word_embedding), dtype=tf.float32)
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
        with tf.variable_scope("basic_lstm"):
            # LSTM layer
            with tf.variable_scope("query_lstm"):
                query_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_layer)
                if self.config.dropout < 1:
                    query_cell = tf.contrib.rnn.DropoutWrapper(
                        query_cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout)
                query_outputs, _ = tf.nn.dynamic_rnn(cell=query_cell,
                                                     inputs=self.queries_embedding,
                                                     sequence_length=self.queries_length,
                                                     dtype=tf.float32)
            with tf.variable_scope("hypothesis_lstm"):
                hypothesis_cell = tf.contrib.rnn.LSTMCell(
                    self.config.hidden_layer)
                if self.config.dropout < 1:
                    hypothesis_cell = tf.contrib.rnn.DropoutWrapper(
                        hypothesis_cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout)
                hypothesis_outputs, _ = tf.nn.dynamic_rnn(cell=hypothesis_cell,
                                                          inputs=self.hypothesis_embedding,
                                                          sequence_length=self.hypothesis_length,
                                                          dtype=tf.float32)

        with tf.variable_scope("match_attention_lstm"):
            # match attention layer
            lstm_m_cell = tf.contrib.rnn.LSTMCell(
                num_units=self.config.hidden_layer)
            if self.config.dropout < 1:
                lstm_m_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_m_cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout)
            We = tf.get_variable(name="We",
                                 shape=[self.config.hidden_layer, 1],
                                 dtype=tf.float32)
            Ws = tf.get_variable(name="Ws",
                                 shape=[self.config.hidden_layer,
                                        self.config.hidden_layer],
                                 dtype=tf.float32)
            Wt = tf.get_variable(name="Wt",
                                 shape=[self.config.hidden_layer,
                                        self.config.hidden_layer],
                                 dtype=tf.float32)
            Wm = tf.get_variable(name="Wm",
                                 shape=[self.config.hidden_layer,
                                        self.config.hidden_layer],
                                 dtype=tf.float32)
            h_s = tf.transpose(query_outputs, [1, 0, 2])

            batch_size, _ = tf.unstack(tf.shape(self.queries))

            #print("hs shape:", h_s.shape)
            def attention_fn_transition(time,  previous_output, previous_state, previous_loop_state):
                if previous_state is None:
                    assert previous_output is None and previous_state is None
                    h_m = tf.zeros([batch_size, self.config.hidden_layer],
                                   dtype=tf.float32, name='PAD')
                    state = lstm_m_cell.zero_state(batch_size, tf.float32)
                    output = None
                else:
                    h_m = previous_output
                    state = previous_state
                    output = previous_output

                #print("h_m shape:", h_m.shape)
                h_t = hypothesis_outputs[:, time, :]
                h_t = tf.reshape(h_t, [-1, self.config.hidden_layer])
                #print("h_t.shape:", h_t.shape)
                e_k = tf.einsum('ijk,kl->ijl',
                                tf.tanh(tf.einsum('ijk,kl->ijl', h_s, Ws) +
                                        tf.matmul(h_t, Wt)) + tf.matmul(h_m, Wm),
                                We)
                alpha_k = tf.nn.softmax(e_k)
                a_k = tf.reduce_sum(tf.multiply(alpha_k, h_s), 0)
                #print("a_k shape:", a_k.shape)
                element_finished = (time >= self.hypothesis_length - 1)
                loop_state = None
                finished = tf.reduce_all(element_finished)
                input = tf.cond(finished,
                                lambda: tf.zeros(
                                    [batch_size, 2 * self.config.hidden_layer], dtype=tf.float32, name='PAD'),
                                lambda: tf.concat([a_k, h_t], 1))

                return (element_finished, input, state, output, loop_state)

            _, (_, attention_final_state), _ = tf.nn.raw_rnn(
                lstm_m_cell, attention_fn_transition)

        with tf.variable_scope("fully_connected"):
            # fully connected classifier
            curr_dim = self.config.hidden_layer
            curr_input = attention_final_state
            Ddim_list = [int(x) for x in self.config.Ddim.split()]
            for i, D in enumerate(Ddim_list):
                w = tf.get_variable(dtype=tf.float32,
                                    shape=[
                                        curr_dim, self.config.hidden_layer * D],
                                    name='w_{}'.format(i))
                b = tf.get_variable(dtype=tf.float32,
                                    shape=[self.config.hidden_layer * D],
                                    name='b_{}'.format(i))
                curr_dim = self.config.hidden_layer * D
                curr_input = tf.matmul(curr_input, w) + b
            w_fc = tf.get_variable(
                shape=[curr_dim, 1], name='w_fc', dtype=tf.float32)
            b_fc = tf.get_variable(shape=[1], name='b_fc', dtype=tf.float32)
            logits = tf.nn.sigmoid(
                tf.matmul(curr_input, w_fc) + b_fc)
            #self.pred_label = tf.nn.softmax(logits) if self.num_class != 1 else  tf.nn.sigmoid(logits)
            self.pred_labels = logits

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
                tf.log(1. + tf.exp(-(self.labels * self.pred_labels - (1 - self.labels) * self.pred_labels)))
            )
