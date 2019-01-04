import tensorflow as tf
#from attention_wrapper import _maybe_mask_score
#from attention_wrapper import *
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.init_ops import Initializer

class identity_initializer(Initializer):
  """Initializer that generates tensors initialized to identity matrix.
  """

  #TODO: for now, the function only works for 2-D matrix.
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return linalg_ops.eye(shape[0], shape[1], dtype=dtype)

  def get_config(self):
      return {"dtype": self.dtype.name}


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

        #match_lstm_cell_attention_fn = lambda curr_input, state: tf.concat(
        #    [curr_input, state], axis=-1)
        
        query_depth = encoded_question.get_shape().as_list()[-1]

        # output attention is false because we want to output the cell output
        # and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            # attention_mechanism_match_lstm = BahdanauAttention(
            #     query_depth, encoded_question, memory_sequence_length=masks_question)
            # cell = tf.contrib.rnn.BasicLSTMCell(
            #     self.hidden_size*2, state_is_tuple=True)
            # lstm_attender = AttentionWrapper(
            #     cell, attention_mechanism_match_lstm,
            #     output_attention=False,
            #     cell_input_fn=match_lstm_cell_attention_fn)

            # # we don't mask the hypothesis because masking the memories will be
            # # handled by the pointerNet
            # reverse_encoded_hypothesis = _reverse(
            #     encoded_hypothesis, masks_hypothesis, 1, 0)

            # output_attender_fw, state_attender_fw = tf.nn.dynamic_rnn(
            #     lstm_attender, encoded_hypothesis, dtype=tf.float32, scope="rnn")
            # output_attender_bw, state_attender_bw = tf.nn.dynamic_rnn(
            #     lstm_attender, reverse_encoded_hypothesis, dtype=tf.float32, scope="rnn")
            # output_attender_bw = _reverse(
            #     output_attender_bw, masks_hypothesis, 1, 0)
            matchlstm_fw_cell = matchLSTMcell(query_depth, self.hidden_size, encoded_question,
                                              masks_question)
            matchlstm_bw_cell = matchLSTMcell(query_depth, self.hidden_size, encoded_question,
                                              masks_question)
            (output_attender_fw, output_attender_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                matchlstm_fw_cell,
                matchlstm_bw_cell,
                encoded_hypothesis,
                sequence_length=masks_hypothesis,
                dtype=tf.float32
            )
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

    def run_answer_ptr(self, output_attender, masks):
        batch_size = tf.shape(output_attender)[0]
        masks_question, masks_passage = masks
        labels = tf.ones([batch_size, 2, 1])


        answer_ptr_cell_input_fn = lambda curr_input, context : context # independent of question
        query_depth_answer_ptr = output_attender.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attender"):
            attention_mechanism_answer_ptr = BahdanauAttention(query_depth_answer_ptr , output_attender, memory_sequence_length = masks_passage)
            # output attention is true because we want to output the attention values
            cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True )
            answer_ptr_attender = AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn = answer_ptr_cell_input_fn)
            logits, _ = tf.nn.static_rnn(answer_ptr_attender, labels, dtype = tf.float32)

        return logits 

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
        logits = self.run_projection(output_maxpool, 1)
        logits_SNLI = self.run_projection(output_maxpool, 3, "SNLI_projection")
        return logits, logits_SNLI


class matchLSTMcell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, h_question, question_m):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question
        # self.question_m = tf.expand_dims(tf.cast(question_m, tf.int32), axis=[2])
        self.question_m = tf.cast(question_m, tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        dtype=tf.float32
        regularizer=None
        with tf.variable_scope(scope):
            # i.e. the batch size
            num_example = tf.shape(self.h_question)[0]
            
            # TODO: figure out the right way to initialize rnn weights.
            # initializer = tf.contrib.layers.xavier_initializer()
            initializer = tf.uniform_unit_scaling_initializer(1.0)

            W_q = tf.get_variable('W_q', [self.input_size, self.input_size], dtype,
                                  initializer, regularizer=regularizer
                                  )
            W_c = tf.get_variable('W_c', [self.input_size, self.input_size], dtype,
                                  initializer, regularizer=regularizer
                                  )
            W_r = tf.get_variable('W_r', [self._state_size, self.input_size], dtype,
                                  # initializer
                                  identity_initializer(), regularizer=regularizer
                                  )
            W_a = tf.get_variable('W_a', [self.input_size, 1], dtype,
                                  initializer, regularizer=regularizer
                                  )
            b_g = tf.get_variable('b_g', [self.input_size], dtype,
                                  tf.zeros_initializer(), regularizer=None)
            b_a = tf.get_variable('b_a', [1], dtype,
                                  tf.zeros_initializer(), regularizer=None)

            wq_e = tf.tile(tf.expand_dims(W_q, axis=[0]), [num_example, 1, 1])
            g = tf.tanh(tf.matmul(self.h_question, wq_e)  # b x q x 2n
                        + tf.expand_dims(tf.matmul(inputs, W_c)
                                         + tf.matmul(state, W_r) + b_g, axis=[1]))
            # TODO:add drop out
            # g = tf.nn.dropout(g, keep_prob=keep_prob)

            wa_e = tf.tile(tf.expand_dims(W_a, axis=0), [num_example, 1, 1])
            # shape: b x q x 1
            a = tf.nn.softmax(tf.squeeze(tf.matmul(g, wa_e) + b_a, axis=[2]))
            # mask out the attention over the padding.
            a = tf.multiply(a, self.question_m)
            question_attend = tf.reduce_sum(tf.multiply(self.h_question, tf.expand_dims(a, axis=[2])), axis=1)

            z = tf.concat([inputs, question_attend], axis=1)

            # NOTE: replace the lstm with GRU.
            # we choose to initialize weight matrix related to hidden to hidden collection with
            # identity initializer.
            W_f = tf.get_variable('W_f', (self._state_size, self._state_size), dtype,
                                  # initializer
                                  identity_initializer(), regularizer=regularizer
                                  )
            U_f = tf.get_variable('U_f', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            # initialize b_f with constant 1.0
            b_f = tf.get_variable('b_f', (self._state_size,), dtype,
                                  tf.constant_initializer(1.0),
                                  regularizer=None)
            W_z = tf.get_variable('W_z', (self.state_size, self._state_size), dtype,
                                  # initializer
                                  identity_initializer(), regularizer=regularizer
                                  )
            U_z = tf.get_variable('U_z', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            # initialize b_z with constant 1.0
            b_z = tf.get_variable('b_z', (self.state_size,), dtype,
                                  tf.constant_initializer(1.0),
                                  regularizer=None)  # tf.zeros_initializer())
            W_o = tf.get_variable('W_o', (self.state_size, self._state_size), dtype,
                                  # initializer
                                  identity_initializer, regularizer=regularizer
                                  )
            U_o = tf.get_variable('U_o', (2 * self.input_size, self._state_size), dtype,
                                  initializer, regularizer=regularizer
                                  )
            b_o = tf.get_variable('b_o', (self._state_size,), dtype,
                                  tf.constant_initializer(0.0), regularizer=None)

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z)
                                + tf.matmul(state, W_z) + b_z)
            f_t = tf.nn.sigmoid(tf.matmul(z, U_f)
                                + tf.matmul(state, W_f) + b_f)
            o_t = tf.nn.tanh(tf.matmul(z, U_o)
                             + tf.matmul(f_t * state, W_o) + b_o)

            output = z_t * state + (1 - z_t) * o_t
            new_state = output

        return output, new_state
