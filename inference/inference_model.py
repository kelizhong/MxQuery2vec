import mxnet as mx
from network.rnn.LSTM import lstm, LSTMParam, LSTMState
from network.seq2seq.encoder import BiDirectionalLstmEncoder


def initial_state_symbol(t_num_lstm_layer, t_num_hidden):
    encoded = mx.sym.Variable("encoded")
    init_weight = mx.sym.Variable("target_init_weight")
    init_bias = mx.sym.Variable("target_init_bias")
    init_h = mx.sym.FullyConnected(data=encoded, num_hidden=t_num_hidden * t_num_lstm_layer,
                                   weight=init_weight, bias=init_bias, name='init_fc')
    init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=t_num_lstm_layer)
    return init_hs


class BiS2SInferenceModel_mask(object):
    def __init__(self,
                 s_num_lstm_layer, s_seq_len, s_vocab_size, s_num_hidden, s_num_embed, s_dropout,
                 t_num_lstm_layer, t_vocab_size, t_num_hidden, t_num_embed, t_num_label, t_dropout,
                 arg_params,
                 use_masking, ctx=mx.cpu(),
                 batch_size=1):
        self.encode_sym = bidirectional_encode_symbol(s_num_lstm_layer, s_seq_len, use_masking,
                                                      s_vocab_size, s_num_hidden, s_num_embed,
                                                      s_dropout)
        self.decode_sym = lstm_decode_symbol(t_num_lstm_layer, t_vocab_size, t_num_hidden,
                                             t_num_embed,
                                             t_num_label, t_dropout)
        self.init_state_sym = initial_state_symbol(t_num_lstm_layer, t_num_hidden)

        # initialize states for LSTM
        forward_source_init_c = [('forward_source_l%d_init_c' % l, (batch_size, s_num_hidden)) for l in
                                 range(s_num_lstm_layer)]
        forward_source_init_h = [('forward_source_l%d_init_h' % l, (batch_size, s_num_hidden)) for l in
                                 range(s_num_lstm_layer)]
        backward_source_init_c = [('backward_source_l%d_init_c' % l, (batch_size, s_num_hidden)) for l in
                                  range(s_num_lstm_layer)]
        backward_source_init_h = [('backward_source_l%d_init_h' % l, (batch_size, s_num_hidden)) for l in
                                  range(s_num_lstm_layer)]
        source_init_states = forward_source_init_c + forward_source_init_h + backward_source_init_c + backward_source_init_h

        target_init_c = [('target_l%d_init_c' % l, (batch_size, t_num_hidden)) for l in range(t_num_lstm_layer)]
        target_init_h = [('target_l%d_init_h' % l, (batch_size, t_num_hidden)) for l in range(t_num_lstm_layer)]
        target_init_states = target_init_c + target_init_h

        encode_data_shape = [("source", (batch_size, s_seq_len))]
        mask_data_shape = [("source_mask", (batch_size, s_seq_len))]
        decode_data_shape = [("target", (batch_size,))]
        init_state_shapes = [("encoded", (batch_size, s_num_hidden * 2))]
        init_last_state_shapes = [("last_encoded", (batch_size, s_num_hidden * 2))]
        encode_input_shapes = dict(source_init_states + encode_data_shape + mask_data_shape)
        decode_input_shapes = dict(target_init_states + decode_data_shape + init_last_state_shapes)
        init_input_shapes = dict(init_state_shapes)
        self.encode_executor = self.encode_sym.simple_bind(ctx=ctx, grad_req='null', **encode_input_shapes)
        self.decode_executor = self.decode_sym.simple_bind(ctx=ctx, grad_req='null', **decode_input_shapes)
        self.init_state_executor = self.init_state_sym.simple_bind(ctx=ctx, grad_req='null', **init_input_shapes)

        for key in self.encode_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encode_executor.arg_dict[key])
        for key in self.decode_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decode_executor.arg_dict[key])
        for key in self.init_state_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.init_state_executor.arg_dict[key])

        encode_state_name = []
        for i in range(s_num_lstm_layer):
            encode_state_name.append("forward_source_l%d_init_c" % i)
            encode_state_name.append("forward_source_l%d_init_h" % i)
            encode_state_name.append("backward_source_l%d_init_c" % i)
            encode_state_name.append("backward_source_l%d_init_h" % i)

        self.encode_state_name = encode_state_name

    def encode(self, input_data, input_mask):
        for key in self.encode_state_name:
            self.encode_executor.arg_dict[key][:] = 0.
        input_data.copyto(self.encode_executor.arg_dict["source"])
        input_mask.copyto(self.encode_executor.arg_dict["source_mask"])
        self.encode_executor.forward()
        last_encoded = self.encode_executor.outputs[0]
        all_encoded = self.encode_executor.outputs[1]
        return last_encoded, all_encoded

    def decode_forward(self, last_encoded, input_data, new_seq):
        if new_seq:
            last_encoded.copyto(self.init_state_executor.arg_dict["encoded"])

            self.init_state_executor.forward()
            # TO-DO multi layer
            init_hs = self.init_state_executor.outputs[0]
            init_hs.copyto(self.decode_executor.arg_dict["target_l0_init_h"])
            self.decode_executor.arg_dict["target_l0_init_c"][:] = 0.0
        last_encoded.copyto(self.decode_executor.arg_dict["last_encoded"])
        input_data.copyto(self.decode_executor.arg_dict["target"])
        self.decode_executor.forward()

        prob = self.decode_executor.outputs[0].asnumpy()

        self.decode_executor.outputs[-3].copyto(self.decode_executor.arg_dict["target_l0_init_c"])
        self.decode_executor.outputs[-2].copyto(self.decode_executor.arg_dict["target_l0_init_h"])
        return prob


def bidirectional_encode_symbol(s_num_lstm_layer, s_seq_len, use_masking, s_vocab_size, s_num_hidden, s_num_embed,
                                s_dropout):
    embed_weight = mx.sym.Variable("embed_weight")
    encoder = BiDirectionalLstmEncoder(seq_len=s_seq_len, use_masking=use_masking, hidden_unit_num=s_num_hidden,
                                       vocab_size=s_vocab_size, embed_size=s_num_embed,
                                       dropout=s_dropout, layer_num=s_num_lstm_layer, embed_weight=embed_weight)
    forward_hidden_all, backward_hidden_all, bi_hidden_all, masks_sliced = encoder.encode()
    concat_encoded = mx.sym.Concat(*bi_hidden_all, dim=1)
    encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='encoded_for_init_state')
    return mx.sym.Group([encoded_for_init_state, concat_encoded])


def lstm_decode_symbol(t_num_lstm_layer, t_vocab_size, t_num_hidden, t_num_embed, t_num_label,
                       t_dropout):
    data = mx.sym.Variable("target")
    seqidx = 0

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("target_cls_weight")
    cls_bias = mx.sym.Variable("target_cls_bias")

    input_weight = mx.sym.Variable("target_input_weight")
    input_bias = mx.sym.Variable("target_input_bias")
    last_encoded = mx.sym.Variable("last_encoded")

    param_cells = []
    last_states = []

    for i in range(t_num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("target_l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("target_l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("target_l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("target_l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
                          h=mx.sym.Variable("target_l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == t_num_lstm_layer)

    hidden = mx.sym.Embedding(data=data,
                              input_dim=t_vocab_size + 1,
                              output_dim=t_num_embed,
                              weight=embed_weight,
                              name="embed_weight")
    con = mx.sym.Concat(hidden, last_encoded)
    hidden = mx.sym.FullyConnected(data=con, num_hidden=t_num_embed,
                                   weight=input_weight, bias=input_bias, name='input_fc')
    # stack LSTM
    for i in range(t_num_lstm_layer):
        if i == 0:
            dp = 0.
        else:
            dp = t_dropout
        next_state = lstm(t_num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state

    fc = mx.sym.FullyConnected(data=hidden, num_hidden=t_num_label,
                               weight=cls_weight, bias=cls_bias, name='target_pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='target_softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
