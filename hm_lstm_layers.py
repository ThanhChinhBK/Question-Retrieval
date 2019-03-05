from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf
import collections


HMLSTMState = collections.namedtuple('HMLSTMCellState', ['c', 'h', 'z'])


class HMLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, batch_size, h_below_size, h_above_size,
                 reuse):
        super(HMLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._h_below_size = h_below_size
        self._h_above_size = h_above_size
        self._batch_size = batch_size

    @property
    def state_size(self):
        # the state is c, h, and z
        return (self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        # outputs h and z
        return self._num_units + 1

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units])
        h = tf.zeros([batch_size, self._num_units])
        z = tf.zeros([batch_size])
        return HMLSTMState(c=c, h=h, z=z)

    def call(self, inputs, state):
        """
        Hierarchical multi-scale long short-term memory cell (HMLSTM)
        inputs: [B, hb_l + 1 + ha_l]
        state: (c=[B, h_l], h=[B, h_l], z=[B, 1])
        output: [B, h_l + 1]
        new_state: (c=[B, h_l], h=[B, h_l], z=[B, 1])
        """
        c = state.c                 # [B, h_l]
        h = state.h                 # [B, h_l]
        z = state.z                 # [B, 1]

        in_splits = tf.constant([self._h_below_size, 1, self._h_above_size])

        hb, zb, ha = array_ops.split(
            value=inputs,
            num_or_size_splits=in_splits,
            axis=1,
            name='split')           # [B, hb_l], [B, 1], [B, ha_l]

        s_recurrent = h             # [B, h_l]

        expanded_z = z              # [B, 1]
        s_above = tf.multiply(expanded_z, ha)   # [B, ha_l]
        s_below = tf.multiply(zb, hb)           # [B, hb_l]

        length = 4 * self._num_units + 1
        states = [s_recurrent, s_above, s_below]

        bias_init = tf.constant_initializer(-1e5, dtype=tf.float32)
        # [B, 4 * h_l + 1]
        concat = rnn_cell_impl._linear(states, length, bias=False,
                                       bias_initializer=bias_init)

        gate_splits = tf.constant(
            ([self._num_units] * 4) + [1], dtype=tf.int32)

        i, g, f, o, z_tilde = array_ops.split(
            value=concat, num_or_size_splits=gate_splits, axis=1)

        i = tf.sigmoid(i)           # [B, h_l]
        g = tf.tanh(g)              # [B, h_l]
        f = tf.sigmoid(f)           # [B, h_l]
        o = tf.sigmoid(o)           # [B, h_l]

        new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
        new_h = self.calculate_new_hidden_state(h, o, new_c, z, zb)
        new_z = tf.expand_dims(self.calculate_new_indicator(z_tilde), -1)

        output = array_ops.concat((new_h, new_z), axis=1)   # [B, h_l + 1]
        new_state = HMLSTMState(c=new_c, h=new_h, z=new_z)

        return output, new_state

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        '''
        update c and h according to correct operations
        c, g, i, f: [B, h_l]
        z, zb: [B, 1]
        new_c: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])                           # [B]
        zb = tf.squeeze(zb, axis=[1])                         # [B]
        new_c = tf.where(
            tf.equal(z, tf.constant(1., dtype=tf.float32)),   # [B]
            tf.multiply(i, g, name='c'),                      # [B, h_l], flush
            tf.where(
                tf.equal(zb, tf.constant(0., dtype=tf.float32)),    # [B]
                tf.identity(c),                               # [B, h_l], copy
                tf.add(tf.multiply(f, c), tf.multiply(i, g))  # [B, h_l], update
            )
        )
        return new_c  # [B, h_l]

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        '''
        h, o, new_c: [B, h_l]
        z, zb: [B, 1]
        new_h: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])             # [B]
        zb = tf.squeeze(zb, axis=[1])           # [B]
        new_h = tf.where(
            tf.logical_and(
                tf.equal(z, tf.constant(0., dtype=tf.float32)),
                tf.equal(zb, tf.constant(0., dtype=tf.float32))
            ),                                  # [B]
            tf.identity(h),                     # [B, h_l], if copy
            tf.multiply(o, tf.tanh(new_c))      # [B, h_l], otherwise
        )
        return new_h                            # [B, h_l]

    def calculate_new_indicator(self, z_tilde):
        # use slope annealing trick
        slope_multiplier = 1  # NOTE: Change this for some tasks
        sigmoided = tf.sigmoid(z_tilde * slope_multiplier)

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(sigmoided, name=name)

        return tf.squeeze(new_z, axis=1)


class MultiHMLSTMCell(rnn_cell_impl.RNNCell):
    """HMLSTM cell composed squentially of individual HMLSTM cells."""

    def __init__(self, cells, reuse):
        super(MultiHMLSTMCell, self).__init__(_reuse=reuse)
        self._cells = cells

    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self._cells]

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state.
        inputs: [B, I + sum(ha_l)]
        state: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        hidden_states: a list of [B, h_l], length L
        new_states: a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]), length L
        """

        total_hidden_size = sum(c._h_above_size for c in self._cells)

        # split out the part of the input that stores values of ha
        raw_inp = inputs[:, :-total_hidden_size]                # [B, I]
        raw_h_aboves = inputs[:, -total_hidden_size:]           # [B, sum(ha_l)]

        ha_splits = [c._h_above_size for c in self._cells]
        h_aboves = array_ops.split(value=raw_h_aboves,
                                   num_or_size_splits=ha_splits, axis=1)

        z_below = tf.ones([tf.shape(inputs)[0], 1])             # [B, 1]
        raw_inp = array_ops.concat([raw_inp, z_below], axis=1)  # [B, I + 1]

        new_states = [0] * len(self._cells)
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                cur_state = state[i]    # ([B, h_l], [B, h_l], [B, 1])
                # i == 0: [B, I + 1] + [B, ha_l] -> [B, I + 1 + ha_l]
                # i != 0: [B, hb_l + 1] + [B, ha_l] -> [B, hb_l + 1 + ha_l]
                cur_inp = array_ops.concat(
                    [raw_inp, h_aboves[i]], axis=1, name='input_to_cell')

                raw_inp, new_state = cell(cur_inp, cur_state)
                new_states[i] = new_state

        hidden_states = [ns.h for ns in new_states]
        return hidden_states, new_states

class HMLSTMNetwork(object):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 out_hidden_size=100,
                 embed_size=100):
        """
        HMLSTMNetwork is a class representing hierarchical multiscale
        long short-term memory network.
        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer correspodning to its index.
        out_hidden_size: integer, the size of the two hidden layers in the
            output network.
        embed_size: integer, the size of the embedding in the output network.
        task: string, one of 'regression' and 'classification'.
        """

        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._session = None
        self._graph = None
        self._task = task
        self._output_size = output_size

        if type(hidden_state_sizes) is list \
            and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the'
                             + ' same as the nubmer of layers.')

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))

        batch_in_shape = (None, None, self._input_size)
        batch_out_shape = (None, None, self._output_size)
        self.batch_in = tf.placeholder(
            tf.float32, shape=batch_in_shape, name='batch_in')
        self.batch_out = tf.placeholder(
            tf.float32, shape=batch_out_shape, name='batch_out')

        self._optimizer = tf.train.AdamOptimizer(1e-3)
        self._initialize_output_variables()
        self._initialize_gate_variables()
        self._initialize_embedding_variables()

    def _initialize_gate_variables(self):
        with vs.variable_scope('gates_vars'):
            for l in range(self._num_layers):
                vs.get_variable(
                    'gate_%s' % l, [sum(self._hidden_state_sizes), 1],
                    dtype=tf.float32)

    def _initialize_embedding_variables(self):
        with vs.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def _initialize_output_variables(self):
        with vs.variable_scope('output_module_vars'):
            vs.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            vs.get_variable('b3', [1, self._output_size], dtype=tf.float32)
            vs.get_variable(
                'w1', [self._embed_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w2', [self._out_hidden_size, self._out_hidden_size],
                dtype=tf.float32)
            vs.get_variable(
                'w3', [self._out_hidden_size, self._output_size],
                dtype=tf.float32)

    def load_variables(self, path='./hmlstm_ckpt'):
        if self._session is None:
            self._session = tf.Session()

            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(self._session, path)

    def save_variables(self, path='./hmlstm_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self._session, path)

    def gate_input(self, hidden_states):
        '''
        gate the incoming hidden states
        hidden_states: [B, sum(h_l)]
        gated_input: [B, sum(h_l)]
        '''
        with vs.variable_scope('gates_vars', reuse=True):
            gates = []  # [[B, 1] for l in range(L)]
            for l in range(self._num_layers):
                weights = vs.get_variable('gate_%d' % l, dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = array_ops.split(
                value=hidden_states,
                num_or_size_splits=self._hidden_state_sizes,
                axis=1)

            gated_list = []  # [[B, h_l] for l in range(L)]
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)  # [B, sum(h_l)]
        return gated_input

    def embed_input(self, gated_input):
        '''
        gated_input: [B, sum(h_l)]
        embedding: [B, E], i.e. [B, embed_size]
        '''
        with vs.variable_scope('embedding_vars', reuse=True):
            embed_weights = vs.get_variable('embed_weights', dtype=tf.float32)

            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

        return embedding

    def output_module(self, embedding, outcome):
        '''
        embedding: [B, E]
        outcome: [B, output_size]
        loss: [B, output_size] or [B, 1]
        prediction: [B, output_size]
        '''
        with vs.variable_scope('output_module_vars', reuse=True):
            b1 = vs.get_variable('b1')
            b2 = vs.get_variable('b2')
            b3 = vs.get_variable('b3')
            w1 = vs.get_variable('w1')
            w2 = vs.get_variable('w2')
            w3 = vs.get_variable('w3')

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # the loss function used below
            # softmax_cross_entropy_with_logits
            prediction = tf.add(tf.matmul(l2, w3), b3, name='prediction')

            loss_args = {'logits': prediction, 'labels': outcome}
            loss = self._loss_function(**loss_args)

            if self._task == 'classification':
                # due to nature of classification loss function
                loss = tf.expand_dims(loss, -1)

        return loss, prediction

    def create_multicell(self, batch_size, reuse):
        def hmlstm_cell(layer):
            if layer == 0:
                h_below_size = self._input_size
            else:
                h_below_size = self._hidden_state_sizes[layer - 1]

            if layer == self._num_layers - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                h_above_size = self._hidden_state_sizes[0]
            else:
                h_above_size = self._hidden_state_sizes[layer + 1]

            return HMLSTMCell(self._hidden_state_sizes[layer], batch_size,
                              h_below_size, h_above_size, reuse)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(self._num_layers)], reuse)

        return hmlstm

    def split_out_cell_states(self, accum):
        '''
        accum: [B, H], i.e. [B, sum(h_l) * 2 + num_layers]
        cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        '''
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]

        split_states = array_ops.split(value=accum,
                                       num_or_size_splits=splits, axis=1)

        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))

        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        '''
        hidden_states: [[B, h_l] for l in range(L)]
        h_aboves: [B, sum(ha_l)], ha denotes h_above
        '''
        concated_hs = array_ops.concat(hidden_states[1:], axis=1)

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1)

        return h_aboves

    def network(self, reuse):
        batch_size = tf.shape(self.batch_in)[1]
        hmlstm = self.create_multicell(batch_size, reuse)

        def scan_rnn(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
            cell_states = self.split_out_cell_states(accum)

            h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)    # [B, sum(ha_l)]
            # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
            hmlstm_in = array_ops.concat((elem, h_aboves), axis=1)
            _, state = hmlstm(hmlstm_in, cell_states)
            # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]) ->
            # a list of [B, h_l + h_l + 1]
            concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
            return array_ops.concat(concated_states, axis=1)    # [B, H]
        # denote 'elem_len' as 'H'
        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        initial = tf.zeros([batch_size, elem_len])              # [B, H]

        states = tf.scan(scan_rnn, self.batch_in, initial)      # [T, B, H]

        def map_indicators(elem):
            state = self.split_out_cell_states(elem)
            return tf.concat([l.z for l in state], axis=1)

        raw_indicators = tf.map_fn(map_indicators, states)      # [T, B, L]
        indicators = tf.transpose(raw_indicators, [1, 2, 0])    # [B, L, T]
        to_map = tf.concat((states, self.batch_out), axis=2)    # [T, B, H + O]

        def map_output(elem):
            splits = tf.constant([elem_len, self._output_size])
            cell_states, outcome = array_ops.split(value=elem,
                                                   num_or_size_splits=splits,
                                                   axis=1)

            hs = [s.h for s in self.split_out_cell_states(cell_states)]
            gated = self.gate_input(tf.concat(hs, axis=1))      # [B, sum(h_l)]
            embeded = self.embed_input(gated)                   # [B, E]
            loss, prediction = self.output_module(embeded, outcome)
            # [B, output_size * 2] or [B, 1 + output_size]
            return tf.concat((loss, prediction), axis=1)

        mapped = tf.map_fn(map_output, to_map)                  # [T, B, _]

        # mapped has diffenent shape for task 'regression' and 'classification'
        loss = tf.reduce_mean(mapped[:, :, :-self._output_size])  # scalar
        predictions = mapped[:, :, -self._output_size:]
        train = self._optimizer.minimize(loss)

        return train, loss, indicators, predictions
