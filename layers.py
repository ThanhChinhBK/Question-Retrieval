import tensorflow as tf
from attention_wrapper import _maybe_mask_score
from attention_wrapper import *
#from tensorflow.contrib.seq2seq import AttentionWrapper
#from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import core as layers_core
import collections

def get_hidden_state(cell_state):
  """ Get the hidden state needed in cell state which is 
      possibly returned by LSTMCell, GRUCell, RNNCell or MultiRNNCell.
  
  Args:
    cell_state: a structure of cell state
  Returns:
    hidden_state: A Tensor
  """

  if type(cell_state) is tuple:
    cell_state = cell_state[-1]
  if hasattr(cell_state, "h"):
    hidden_state = cell_state.h
  else:
    hidden_state = cell_state
  return hidden_state


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
        masks_question = tf.reduce_sum(masks_question, -1)
        masks_hypothesis = tf.reduce_sum(masks_hypothesis, -1)

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

    def run_match_lstm(self, encoded_rep, masks, return_sequence=False):
        encoded_question, encoded_hypothesis = encoded_rep
        masks_question, masks_hypothesis = masks
        masks_hypothesis = tf.reduce_sum(masks_hypothesis, -1)
        #masks_question = tf.reduce_sum(masks_question, -1)
        match_lstm_cell_attention_fn = lambda curr_input, state: tf.concat(
            [curr_input, state], axis=-1)
        
        query_depth = encoded_question.get_shape().as_list()[-1]

        # output attention is false because we want to output the cell output
        # and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_fw = SeqMatchSeqAttention(query_depth, encoded_question, masks_question)
            mLSTM_fw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
                                               attention_mechanism_fw)
            attention_mechanism_bw = SeqMatchSeqAttention(query_depth, encoded_question, masks_question)
            mLSTM_bw_cell = SeqMatchSeqWrapper(tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True),
                                               attention_mechanism_bw)
            (output_attender_fw, output_attender_bw), (state_attender_fw, state_attender_bw) = tf.nn.bidirectional_dynamic_rnn(
              mLSTM_fw_cell,
              mLSTM_bw_cell,
              encoded_hypothesis,
              sequence_length=masks_hypothesis,
              dtype=tf.float32
             )
            
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
            #      output_attender_bw, masks_hypothesis, 1, 0)
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


class SeqMatchSeqAttentionState(
    collections.namedtuple("SeqMatchSeqAttentionState", ("cell_state", "attention"))):
  pass

class SeqMatchSeqAttention(object):
  """ Attention for SeqMatchSeq.
  """

  def __init__(self,num_units,premise_mem,premise_mem_weights,name="SeqMatchSeqAttention"):
    """ Init SeqMatchSeqAttention
    Args:
      num_units: The depth of the attention mechanism.
      premise_mem: encoded premise memory
      premise_mem_weights: premise memory weights
    """
    # Init layers
    self._name = name
    self._num_units = num_units
    # Shape: [batch_size,max_premise_len,rnn_size]
    self._premise_mem = premise_mem
    # Shape: [batch_size,max_premise_len]
    self._premise_mem_weights = premise_mem_weights

    with tf.name_scope(self._name):
      self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
      self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
      self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
      # Preprocess premise Memory
      # Shape: [batch_size, max_premise_len, num_units]
      self._keys = self.premise_mem_layer(premise_mem)
      self.batch_size = self._keys.shape[0].value 
      self.alignments_size = self._keys.shape[1].value 

  def __call__(self, hypothesis_mem, query):
    """ Perform attention
    Args:
      hypothesis_mem: hypothesis memory
      query: hidden state from last time step
    Returns:
      attention: computed attention
    """
    with tf.name_scope(self._name):
      # Shape: [batch_size, 1, num_units]
      processed_hypothesis_mem = tf.expand_dims(self.hypothesis_mem_layer(hypothesis_mem), 1)
      # Shape: [batch_size, 1, num_units]
      processed_query = tf.expand_dims(self.query_layer(query), 1)
      v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
      # Shape: [batch_size, max_premise_len]
      score = tf.reduce_sum(v * tf.tanh(self._keys + processed_hypothesis_mem + processed_query), [2])
      # Mask score with -inf
      score_mask_values = float("-inf") * (1.-tf.cast(self._premise_mem_weights, tf.float32))
      masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
      # Calculate alignments
      # Shape: [batch_size, max_premise_len]
      alignments = tf.nn.softmax(masked_score)
      # Calculate attention
      # Shape: [batch_size, rnn_size]
      attention = tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1)
      return attention


class SeqMatchSeqWrapper(rnn_cell_impl.RNNCell):
  """ RNN Wrapper for SeqMatchSeq.
  """
  def __init__(self, cell, attention_mechanism, name='SeqMatchSeqWrapper'):
    super(SeqMatchSeqWrapper, self).__init__(name=name)
    self._cell = cell
    self._attention_mechanism = attention_mechanism

  def call(self, inputs, state):
    """
    Args:
      inputs: inputs at some time step
      state: A (structure of) cell state
    """
    # Concatenate attention and input 
    cell_inputs = tf.concat([state.attention, inputs], axis=-1)
    cell_state = state.cell_state
    # Call cell function
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
    # Get hidden state
    hidden_state = get_hidden_state(cell_state)
    # Calculate attention
    attention = self._attention_mechanism(inputs, hidden_state)
    # Assemble next state
    next_state = SeqMatchSeqAttentionState(
      cell_state=next_cell_state,
      attention=attention)
    return cell_output, next_state

  @property
  def state_size(self):
    return SeqMatchSeqAttentionState(
        cell_state=self._cell.state_size,
        attention=self._attention_mechanism._premise_mem.get_shape()[-1].value
        )

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    cell_state = self._cell.zero_state(batch_size, dtype)
    attention = rnn_cell_impl._zero_state_tensors(self.state_size.attention, batch_size, tf.float32)
    return SeqMatchSeqAttentionState(
          cell_state=cell_state,
      attention=attention)
