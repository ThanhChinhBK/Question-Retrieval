import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, maxpooling_classifier


class Rnet(object):

    def __init__(self, config, vocab, word_embedding):
        self.config = config
        self.vocab = vocab
        self.word_embedding = word_embedding
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.add_placeholder()
        self.ready()


        self.lr = tf.get_variable(
            "lr",
            dtype=tf.float32, trainable=False,
            initializer=tf.constant(self.config.learning_rate)
        )
        #self.train_op = tf.train.AdamOptimizer(
        #    learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, config.grad_clip)
        self.train_op = self.opt.apply_gradients(
            zip(capped_grads, variables), global_step=self.global_step)

    def add_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.queries = tf.placeholder(
                tf.int32, [None, self.config.pad], "queries")
            self.queries_length = tf.placeholder(
                tf.int32, [None], "queries_length")
            self.hypothesis = tf.placeholder(
                tf.int32, [None, self.config.pad], "hypothesis")
            self.hypothesis_length = tf.placeholder(
                tf.int32, [None], "hypothesis_length")
            self.y = tf.placeholder(
                tf.float32, [None], "labels")
            self.dropout = tf.placeholder(
                tf.float32, [], "dropout")

    def ready(self):
        config = self.config
        self.q_mask = tf.cast(self.queries, tf.bool)
        self.h_mask = tf.cast(self.hypothesis, tf.bool)
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            # with tf.variable_scope("char"):
            #     ch_emb = tf.reshape(tf.nn.embedding_lookup(
            #         self.char_mat, self.ch), [N * PL, CL, dc])
            #     qh_emb = tf.reshape(tf.nn.embedding_lookup(
            #         self.char_mat, self.qh), [N * QL, CL, dc])
            #     ch_emb = dropout(
            #         ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #     qh_emb = dropout(
            #         qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
            #     cell_fw = tf.contrib.rnn.GRUCell(dg)
            #     cell_bw = tf.contrib.rnn.GRUCell(dg)
            #     _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #         cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
            #     ch_emb = tf.concat([state_fw, state_bw], axis=1)
            #     _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #         cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
            #     qh_emb = tf.concat([state_fw, state_bw], axis=1)
            #     qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
            #     ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                init_emb = tf.constant(self.vocab.embmatrix(
                    self.word_embedding), dtype=tf.float32)
                embeddings = tf.get_variable("word_embedding",
                                             initializer=init_emb,
                                             dtype=tf.float32)
                c_emb = tf.nn.embedding_lookup(embeddings, self.hypothesis)
                q_emb = tf.nn.embedding_lookup(embeddings, self.queries)

            #c_emb = tf.concat([c_emb, ch_emb], axis=2)
            #q_emb = tf.concat([q_emb, qh_emb], axis=2)
        batch_size, _ = tf.unstack(tf.shape(self.queries))
        with tf.variable_scope("encoding"):
            rnn = gru(
                num_layers=3,
                num_units=self.config.hidden_layer,
                batch_size=batch_size,
                input_size=c_emb.get_shape().as_list()[-1],
                keep_prob=config.dropout,
                is_train=True
            )
            c = rnn(c_emb, seq_len=self.hypothesis_length)
            q = rnn(q_emb, seq_len=self.queries_length)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=self.config.hidden_layer,
                                   keep_prob=config.dropout, is_train=True)
            rnn = gru(num_layers=1,
                      num_units=self.config.hidden_layer,
                      batch_size=batch_size,
                      input_size=qc_att.get_shape().as_list()[-1],
                      keep_prob=config.dropout,
                      is_train=True)
            att = rnn(qc_att, seq_len=self.hypothesis_length)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.h_mask, hidden=self.config.hidden_layer,
                keep_prob=config.dropout, is_train=True)
            rnn = gru(num_layers=1,
                      num_units=self.config.hidden_layer,
                      batch_size=batch_size,
                      input_size=self_att.get_shape().as_list()[-1],
                      keep_prob=config.dropout,
                      is_train=True)
            match = rnn(self_att, seq_len=self.hypothesis_length)
            
        
        with tf.variable_scope("predict"):
            output = maxpooling_classifier(match)
            self.yp = tf.nn.sigmoid(output)
            print(self.y.get_shape())
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.reshape(self.y,(-1,1)), logits=output))
            vars = tf.trainable_variables()
            print("Number of parameter is {}".format(len(vars)))
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name and "embedd" not in v.name]) * 1e-4
            self.loss = self.loss + lossL2
            

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
