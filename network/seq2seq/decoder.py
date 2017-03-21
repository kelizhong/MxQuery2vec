import mxnet as mx

from network.rnn.lstm import lstm, LSTMParam, LSTMState


class LstmDecoder(object):
    """An attention-based decoder
    Parameters
    ----------
    seq_len: int
        decoder sequence length
    use_masking: bool
        whether use masking
    hidden_unit_num: int
        number of hidden units in the neural network for decoder
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
    attention: attention class
        attention for decoder
    name: str
        decoder name
    """
    def __init__(self, seq_len, use_masking,
                 hidden_unit_num,
                 vocab_size, embed_size,
                 dropout=0.0, layer_num=1,
                 embed_weight=None, attention=None, name='decoder'):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.hidden_unit_num = hidden_unit_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.embed_weight = embed_weight
        self.attention = attention
        self.name = name

    def decode(self, init_state, encoder_hidden_all=None, encoder_mask=None, encoder_hidden_size=None):
        if self.attention:
            all_attended = mx.sym.Concat(*encoder_hidden_all, dim=1, name='concat_attended')  # (batch, n * seq_len)
            all_attended = mx.sym.Reshape(data=all_attended,
                                          shape=(-1, len(encoder_hidden_all), encoder_hidden_size),
                                          name='_reshape_concat_attended')
        data = mx.sym.Variable('decoder_data')  # decoder input data
        label = mx.sym.Variable('decoder_softmax_label')  # decoder label data
        # declare variables
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable("{}_embed_weight".format(self.name))
        cls_weight = mx.sym.Variable("decoder_cls_weight")
        cls_bias = mx.sym.Variable("decoder_cls_bias")
        init_weight = mx.sym.Variable("decoder_init_weight")
        init_bias = mx.sym.Variable("decoder_init_bias")
        input_weight = mx.sym.Variable("decoder_input_weight")

        param_cells = []
        last_states = []
        init_h = mx.sym.FullyConnected(data=init_state, num_hidden=self.hidden_unit_num * self.layer_num,
                                       weight=init_weight, bias=init_bias, name='init_fc')
        init_h = mx.sym.Activation(data=init_h, act_type='tanh', name='init_act')
        init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=self.layer_num)
        for i in range(self.layer_num):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("decoder_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("decoder_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("decoder_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("decoder_l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("decoder_l%d_init_c" % i),
                              h=init_hs[i])
            last_states.append(state)
        assert (len(last_states) == self.layer_num)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_size,
                                 name="{}_embed".format(self.name))
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('decoder_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_decoder_mask')
        hidden_all = []
        for seq_idx in range(self.seq_len):
            if self.attention:
                hidden = self.attention.attend(encoder_hidden_all=encoder_hidden_all, concat_attended=all_attended,
                                                                  state=last_states[0].h,
                                                                  attend_masks=encoder_mask,
                                                                  use_masking=True)
            else:
                hidden = init_state
            con = mx.sym.Concat(wordvec[seq_idx], hidden)
            hidden = mx.sym.FullyConnected(data=con, num_hidden=self.embed_size,
                                           weight=input_weight, no_bias=True, name='input_fc')

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
                    prev_state_c = last_states[i].c
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    new_c = mx.sym.broadcast_mul(1.0 - mask, prev_state_c) + mx.sym.broadcast_mul(mask, next_state.c)
                    next_state = LSTMState(c=new_c, h=new_h)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.vocab_size,
                                     weight=cls_weight, bias=cls_bias, name='decoder_pred')

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))
        if self.use_masking:
            loss_mask = mx.sym.transpose(data=input_mask)
            loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
            pred = mx.sym.broadcast_mul(pred, loss_mask)
        # softmaxwithloss http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='decoder_softmax', ignore_label=0, use_ignore=True,
                                  normalization='valid')

        return sm
