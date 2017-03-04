import mxnet as mx

from network.rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
from network.rnn.GRU import gru, GRUModel, GRUParam, GRUState


class LstmDecoder(object):
    def __init__(self, seq_len, use_masking,
                 hidden_unit_num,
                 vocab_size, embed_size,
                 dropout=0.0, layer_num=1,
                 embed_weight=None, embed_name='embed_weight'):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.hidden_unit_num = hidden_unit_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.embed_weight = embed_weight
        self.embed_name = embed_name

    def decode(self, encoded):
        data = mx.sym.Variable('target')  # target input data
        label = mx.sym.Variable('target_softmax_label')  # target label data
        # declare variables
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable(self.embedding_name)
        cls_weight = mx.sym.Variable("target_cls_weight")
        cls_bias = mx.sym.Variable("target_cls_bias")
        init_weight = mx.sym.Variable("target_init_weight")
        init_bias = mx.sym.Variable("target_init_bias")
        input_weight = mx.sym.Variable("target_input_weight")
        input_bias = mx.sym.Variable("target_input_bias")

        param_cells = []
        last_states = []
        init_h = mx.sym.FullyConnected(data=encoded, num_hidden=self.hidden_unit_num * self.layer_num,
                                       weight=init_weight, bias=init_bias, name='init_fc')
        init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=self.layer_num)
        for i in range(self.layer_num):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("target_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("target_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("target_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("target_l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
                              h=init_hs[i])
            last_states.append(state)
        assert (len(last_states) == self.layer_num)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=self.embed_weight, output_dim=self.embed_size, name=self.embed_name)
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('target_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_target_mask')

        hidden_all = []
        for seq_idx in range(self.seq_len):
            con = mx.sym.Concat(wordvec[seq_idx], encoded)
            hidden = mx.sym.FullyConnected(data=con, num_hidden=self.embed_size,
                                           weight=input_weight, bias=input_bias, name='input_fc')

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

                if self.use_masking:
                    prev_state_h = last_states[i].h
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    next_state = LSTMState(c=next_state.c, h=new_h)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.vocab_size,
                                     weight=cls_weight, bias=cls_bias, name='target_pred')

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))
        if self.use_masking:
            loss_mask = mx.sym.transpose(data=input_mask)
            loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
            pred = mx.sym.broadcast_mul(pred, loss_mask)

        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='target_softmax')

        return sm

