import tensorflow as tf
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

    # TODO: for now, the function only works for 2-D matrix.
    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return linalg_ops.eye(shape[0], shape[1], dtype=dtype)

    def get_config(self):
        return {"dtype": self.dtype.name}


def cnnsum(inputs, dropout, cnnact=tf.nn.relu,
           cdim={1: 1 / 2, 2: 1 / 2, 3: 1 / 2, 4: 1 /
                 2, 5: 1 / 2, 6: 1 / 2, 7: 1 / 2},
           input_dim=300, pad=100, pfx=''):
    si_cnn_res_list = []
    tot_len = 0
    for fl, cd in cdim.items():
        nb_filter = int(input_dim * cd)
        si_conv = tf.layers.conv1d(name=pfx + 'conv%d' % (fl), inputs=inputs,
                                   kernel_size=fl, filters=nb_filter)
        si_cnn_one = cnnact(tf.layers.batch_normalization(si_conv))

        si_pool_one = tf.layers.max_pooling1d(si_cnn_one,
                                              pool_size=int(pad - fl + 1),
                                              strides=int(pad - fl + 1),
                                              name=pfx + 'pool%d' % (fl))

        si_out_one = tf.contrib.layers.flatten(si_pool_one)
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
            #score_mask_values = float("-inf") * (1.-tf.cast(self._premise_mem_weights, tf.float32))
            #masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
            # Calculate alignments
            # Shape: [batch_size, max_premise_len]
            alignments = tf.nn.softmax(score)
            # Calculate attention
            # Shape: [batch_size, rnn_size]
            attention = tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1)
            return attention, alignments
        
        
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
        cell_state = tf.nn.rnn_cell.LSTMStateTuple(c=cell_state.c,
                                                   h=cell_state.h[:, :-self._attention_mechanism.alignments_size])
        # Call cell function
        
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        # Get hidden state
        hidden_state = get_hidden_state(cell_state)
        # Calculate attention
        attention, alignments = self._attention_mechanism(inputs, hidden_state)
        # Assemble next state
        
        next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(c=next_cell_state.c,
                                                        h=tf.concat((next_cell_state.h, alignments), -1))
        next_state = SeqMatchSeqAttentionState(
            cell_state=next_cell_state,
            attention=attention)
        return tf.concat([cell_output, alignments],-1), next_state
    
    @property
    def state_size(self):
        state_size = self._cell.state_size
        state_size = (state_size[0], self._attention_mechanism.alignments_size + state_size[1])
        return SeqMatchSeqAttentionState(
            cell_state=state_size,
            attention=self._attention_mechanism._premise_mem.get_shape()[-1].value
        )
    
    @property
    def output_size(self):
        return self._cell.output_size + self._attention_mechanism.alignments_size
    
    def zero_state(self, batch_size, dtype):
        cell_state = self._cell.zero_state(batch_size, dtype)
        cell_state = tf.nn.rnn_cell.LSTMStateTuple(c=cell_state.c,
                                                   h=tf.concat([cell_state.h,
                                                                rnn_cell_impl._zero_state_tensors(self._attention_mechanism.alignments_size, batch_size, tf.float32)],-1))
        attention = rnn_cell_impl._zero_state_tensors(self.state_size.attention, batch_size, tf.float32)
        return SeqMatchSeqAttentionState(
            cell_state=cell_state,
            attention=attention)
    
class PointerNetLSTMCell(tf.contrib.rnn.LSTMCell):
    """
    Implements the Pointer Network Cell
    """
    
    def __init__(self, num_units, context_to_point, mask):
        super(PointerNetLSTMCell, self).__init__(
            num_units, state_is_tuple=True)
        #seq_pad = tf.shape(context_to_point)[1]
        self.sequence_mask = tf.cast(
            mask, tf.float32)

        self.context_to_point = context_to_point
        self.fc_context = tf.contrib.layers.fully_connected(self.context_to_point,
                                                            num_outputs=self._num_units,
                                                            activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tf.contrib.layers.fully_connected(m_prev,
                                                                           num_outputs=self._num_units,
                                                                           activation_fn=None),
                                         1))
            logits = tf.contrib.layers.fully_connected(
                U, num_outputs=1, activation_fn=None)
            logits = logits * tf.expand_dims(self.sequence_mask, -1)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(
                self.context_to_point * scores, axis=1)
            lstm_out, lstm_state = super(
                PointerNetLSTMCell, self).__call__(attended_context, state)
        return tf.squeeze(scores, -1), lstm_state

def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    print(inputs_len)
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=tf.float32)
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)
        scores, cur_state = cell(cur_x, prev_s)

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)

        if isinstance(cell, tf.contrib.rnn.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tf.contrib.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(
            tf.reduce_all(finished)),
        body=loop_fn,
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state

def self_attentive_encode(hypothesis_length, hidden, dropout, attn_unit, hop):
    batch_size, max_len, hidden_dim = hidden.get_shape().as_list()
    hidden_compressed = tf.reshape(hidden, [-1, hidden_dim]) # [batch_size * max_len, hidden_dim]
    attn_mask = tf.expand_dims(hypothesis_length, 1)  #[batch_size,1 , max_len]
    attn_mask = tf.tile(attn_mask, [1,hop, 1]) # [batch_size, hop, 1]
    ws1 = layers_core.Dense(attn_unit, name="ws1", use_bias=False)
    ws2 = layers_core.Dense(hop, name="ws2", use_bias=False)
    hbar = tf.tanh(ws1(hidden_compressed)) # [batch_size*max_len, attn_unit]
    hbar = tf.nn.dropout(hbar, dropout)
    alphas = tf.transpose(tf.reshape(ws2(hbar), [-1, max_len, hop]),
                          [0,2,1]) #[batch_size, hop, max_len]
    alphas_penalized = float("-inf") * (1. - tf.cast(attn_mask, tf.float32))
    alphas_penalized = tf.where(tf.cast(attn_mask, tf.bool), alphas, alphas_penalized)
    alphas = tf.nn.softmax(alphas) #[batch_size, hop, max_len]
    return tf.matmul(alphas, hidden), alphas #[batch_size, hop, hidden_dim]
