import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import core as layers_core
import collections
import numpy as np


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



class SeqMatchSeq(object):

    def __init__(self, flags, vocab, char_vocab, word_embedding):
        self.config = flags
        self.Ddim = [int(x) for x in self.config.Ddim.split()]
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.word_embedding = word_embedding
        self._add_placeholder()
        
        self._add_embedding()
        self._build_model()


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
        cell = tf.contrib.rnn.LSTMCell
        with tf.variable_scope("premise_encoding"):
            # Create premise encoder with dropout
            premise_encoder = tf.contrib.rnn.DropoutWrapper(cell(self.config.hidden_layer),
                                                            input_keep_prob=1 - self.dropout,
                                                            output_keep_prob=1 - self.dropout)
            # Encode premise
            # Shape: [batch_size, max_time, rnn_size]
            premise_mem,_ = tf.nn.dynamic_rnn(premise_encoder,
                                              self.queries_embedding,
                                              tf.reduce_sum(self.queries_length, -1),
                                              dtype=tf.float32)
        with tf.variable_scope("hypothesis_encoding"):
            # Create hypothesis encoder with dropout
            hypothesis_encoder = tf.contrib.rnn.DropoutWrapper(cell(self.config.hidden_layer),
                                                               input_keep_prob=1 - self.dropout,
                                                               output_keep_prob=1 - self.dropout)
            # Encode hypothesis
            # Shape: [batch_size, max_time, rnn_size]
            hypothesis_mem,_ = tf.nn.dynamic_rnn(hypothesis_encoder,
                                                 self.hypothesis_embedding,
                                                 tf.reduce_sum(self.hypothesis_length, -1),
                                                 dtype=tf.float32)
        # Use SeqMatchSeq Attention Mechanism
        attention_mechanism = SeqMatchSeqAttention(self.config.hidden_layer,
                                                   premise_mem,
                                                   self.queries_length)
        # match LSTM
        mLSTM = cell(self.config.hidden_layer)
        # Wrap mLSTM
        mLSTM = SeqMatchSeqWrapper(mLSTM,attention_mechanism)

        # Training Helper
        #helper = tf.contrib.seq2seq.TrainingHelper(hypothesis_mem, self._hypothesis_lens)    
        # Basic Decoder
        #decoder = tf.contrib.seq2seq.BasicDecoder(mLSTM, helper, mLSTM.zero_state(batch_size,tf.float32)) 
        # Decode
        #_, state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True)

        _, state = tf.nn.dynamic_rnn(mLSTM, hypothesis_mem,
                                     tf.reduce_sum(self.hypothesis_length, -1),
                                     dtype=tf.float32)
        hidden_state = get_hidden_state(state.cell_state)
        # Fully connection Layer
        fcn = layers_core.Dense(3, name='fcn')
        # logits
        logits = fcn(hidden_state)
        
        # predicted_ids_with_logits
        self.yp_SNLI = tf.nn.top_k(logits)
        # Losses
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_SNLI, logits=logits)
        # Total loss
        self.loss_SNLI = tf.reduce_mean(losses)
        # Get all trainable variables
        parameters = tf.trainable_variables()
        # Calculate gradients
        gradients = tf.gradients(self.loss_SNLI, parameters)
        # Clip gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        # Optimization
        #optimizer = tf.train.GradientDescentOptimizer(self.init_learning_rate)
        SNLI_op = tf.train.AdamOptimizer(self.config.learning_rate)
        # Update operator
        self.global_step = tf.get_variable('global_step',shape=[],initializer=tf.constant_initializer(0,dtype=tf.int32),trainable=False)
        self.train_op_SNLI = SNLI_op.apply_gradients(zip(clipped_gradients, parameters),global_step=self.global_step)

