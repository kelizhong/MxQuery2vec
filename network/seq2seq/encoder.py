# -*- coding: utf-8 -*-
import mxnet as mx
from ..rnn.lstm import lstm, LSTMParam, LSTMState


class LstmEncoder(object):
    """ Encoder for seq2seq model
    Parameters
    ----------
    seq_len: int
        encoder sequence length
    use_masking: bool
        whether use masking
    hidden_unit_num: int
        number of hidden units in the neural network for encoder
    vocab_size: int
        vocabulary size
    embed_size: int
        word embedding size
    dropout: float
        the probability to ignore the neuron outputs
    layer_num int
        decoder layer num
    embed_weight: sym.Variable
        word embedding weight
    name: str
        decoder name
    """
    def __init__(self, seq_len, use_masking,
                 hidden_unit_num,
                 vocab_size, embed_size,
                 dropout=0.0, layer_num=1, embed_weight=None, name='encoder'):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.hidden_unit_num = hidden_unit_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.embed_weight = embed_weight
        self.name = name

    def encode(self):
        data = mx.sym.Variable('encoder_data')  # input data, source

        # word embedding weight
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable("{}_embed_weight".format(self.name))

        param_cells = []
        last_states = []
        for i in range(self.layer_num):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("encoder_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("encoder_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("encoder_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("encoder_l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("encoder_l%d_init_c" % i),
                              h=mx.sym.Variable("encoder_l%d_init_h" % i))
            last_states.append(state)
        assert (len(last_states) == self.layer_num)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_size,
                                 name="{}_embed".format(self.name))
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)

        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('encoder_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_encoder_mask')

        hidden_all = []
        for seq_idx in range(self.seq_len):
            hidden = wordvec[seq_idx]
            if self.use_masking:
                mask = masks[seq_idx]

            # stack LSTM
            for i in range(self.layer_num):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = lstm(self.hidden_unit_num, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seq_idx, layeridx=i, dropout=dp_ratio)

                # process masking
                if self.use_masking:
                    prev_state_h = last_states[i].h
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    next_state = LSTMState(c=next_state.c, h=new_h)

                hidden = next_state.h
                last_states[i] = next_state

            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        return hidden_all


class BiDirectionalLstmEncoder(object):
    def __init__(self, seq_len, use_masking,
                 hidden_unit_num,
                 vocab_size, embed_size,
                 dropout=0.0, layer_num=1, embed_weight=None, name='encoder'):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.hidden_unit_num = hidden_unit_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.embed_weight = embed_weight
        self.name = name

    def encode(self):
        # declare variables
        data = mx.sym.Variable('encoder_data')  # input data, encoder for encoder

        # word embedding weight
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable("{}_embed_weight".format(self.name))

        # encoder bi-lstm parameters
        forward_param_cells = []
        forward_last_states = []
        for i in range(self.layer_num):
            forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("forward_encoder_l%d_i2h_weight" % i),
                                                 i2h_bias=mx.sym.Variable("forward_encoder_l%d_i2h_bias" % i),
                                                 h2h_weight=mx.sym.Variable("forward_encoder_l%d_h2h_weight" % i),
                                                 h2h_bias=mx.sym.Variable("forward_encoder_l%d_h2h_bias" % i)))
            forward_state = LSTMState(c=mx.sym.Variable("forward_encoder_l%d_init_c" % i),
                                      h=mx.sym.Variable("forward_encoder_l%d_init_h" % i))
            forward_last_states.append(forward_state)
        assert (len(forward_last_states) == self.layer_num)
        backward_param_cells = []
        backward_last_states = []
        for i in range(self.layer_num):
            backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("backward_encoder_l%d_i2h_weight" % i),
                                                  i2h_bias=mx.sym.Variable("backward_encoder_l%d_i2h_bias" % i),
                                                  h2h_weight=mx.sym.Variable("backward_encoder_l%d_h2h_weight" % i),
                                                  h2h_bias=mx.sym.Variable("backward_encoder_l%d_h2h_bias" % i)))
            backward_state = LSTMState(c=mx.sym.Variable("backward_encoder_l%d_init_c" % i),
                                       h=mx.sym.Variable("backward_encoder_l%d_init_h" % i))
            backward_last_states.append(backward_state)
        assert (len(backward_last_states) == self.layer_num)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_size,
                                 name="{}_embed".format(self.name))
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)

        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('encoder_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_encoder_mask')

        forward_hidden_all = []
        backward_hidden_all = []
        for seq_idx in xrange(self.seq_len):
            forward_hidden = wordvec[seq_idx]
            backward_hidden = wordvec[self.seq_len - 1 - seq_idx]
            if self.use_masking:
                forward_mask = masks[seq_idx]
                backward_mask = masks[self.seq_len - 1 - seq_idx]

            # stack LSTM
            for i in xrange(self.layer_num):
                if i == 0:
                    dropout = 0.
                else:
                    dropout = self.dropout
                forward_next_state = lstm(self.hidden_unit_num, indata=forward_hidden,
                                          prev_state=forward_last_states[i],
                                          param=forward_param_cells[i],
                                          seqidx=seq_idx, layeridx=i, dropout=dropout)
                backward_next_state = lstm(self.hidden_unit_num, indata=backward_hidden,
                                           prev_state=backward_last_states[i],
                                           param=backward_param_cells[i],
                                           seqidx=seq_idx, layeridx=i, dropout=dropout)

                # process masking https://github.com/dmlc/mxnet/issues/2401
                if self.use_masking:
                    forward_prev_state_h = forward_last_states[i].h
                    forward_prev_state_c = forward_last_states[i].c
                    forward_new_h = mx.sym.broadcast_mul(1.0 - forward_mask,
                                                         forward_prev_state_h) + mx.sym.broadcast_mul(
                        forward_mask,
                        forward_next_state.h)
                    forward_new_c = mx.sym.broadcast_mul(1.0 - forward_mask,
                                                         forward_prev_state_c) + mx.sym.broadcast_mul(
                        forward_mask,
                        forward_next_state.c)
                    forward_next_state = LSTMState(c=forward_new_c, h=forward_new_h)

                    backward_prev_state_h = backward_last_states[i].h
                    backward_prev_state_c = backward_last_states[i].c
                    backward_new_h = mx.sym.broadcast_mul(1.0 - backward_mask,
                                                          backward_prev_state_h) + mx.sym.broadcast_mul(
                        backward_mask,
                        backward_next_state.h)
                    backward_new_c = mx.sym.broadcast_mul(1.0 - backward_mask,
                                                          backward_prev_state_c) + mx.sym.broadcast_mul(
                        backward_mask,
                        backward_next_state.c)
                    backward_next_state = LSTMState(c=backward_new_c, h=backward_new_h)

                forward_hidden = forward_next_state.h
                forward_last_states[i] = forward_next_state
                backward_hidden = backward_next_state.h
                backward_last_states[i] = backward_next_state

            if self.dropout > 0.:
                forward_hidden = mx.sym.Dropout(data=forward_hidden, p=self.dropout)
                backward_hidden = mx.sym.Dropout(data=backward_hidden, p=self.dropout)

            forward_hidden_all.append(forward_hidden)
            backward_hidden_all.insert(0, backward_hidden)

        bi_hidden_all = []
        for f, b in zip(forward_hidden_all, backward_hidden_all):
            bi = mx.sym.Concat(f, b, dim=1)
            bi_hidden_all.append(bi)

        if self.use_masking:
            return forward_hidden_all, backward_hidden_all, bi_hidden_all, masks
        else:
            return forward_hidden_all, backward_hidden_all, bi_hidden_all
