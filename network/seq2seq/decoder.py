# coding=utf-8
# pylint: disable=import-error, too-many-arguments
"""decoder with lstm cell"""
import mxnet as mx

from network.rnn.lstm import lstm, LSTMState

from base.decoder import Decoder

# pylint: disable=pointless-string-statement
'''
Papers:
[1] Learning Phrase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation (http://arxiv.org/abs/1406.1078)
'''


class LstmDecoder(Decoder):
    """Lstm decoder, accept the encoder to init the state. Implementation base on[1]
        y(t) = LSTM(s(t-1), y(t-1), C); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder
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
        name: str
            decoder name
    """

    def __init__(self, seq_len, use_masking,
                 hidden_unit_num,
                 vocab_size, embed_size,
                 dropout=0.0, layer_num=1,
                 embed_weight=None, name='decoder'):
        super(LstmDecoder, self).__init__(seq_len, use_masking,
                                          hidden_unit_num,
                                          vocab_size, embed_size,
                                          dropout=dropout, layer_num=layer_num,
                                          embed_weight=embed_weight, name=name)

    # pylint: disable=too-many-locals
    def decode(self, init_state):

        param_cells, last_states = self.init_cell_state_parameter(init_state)
        data = mx.sym.Variable('decoder_data')  # decoder input data
        label = mx.sym.Variable('decoder_softmax_label')  # decoder label data

        cls_weight = mx.sym.Variable("decoder_cls_weight")
        cls_bias = mx.sym.Variable("decoder_cls_bias")

        input_weight = mx.sym.Variable("decoder_input_weight")

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_size,
                                 name="{}_embed".format(self.name))
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('decoder_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len,
                                        name='sliced_decoder_mask')
        hidden_all = []
        for seq_id in range(self.seq_len):
            con = mx.sym.Concat(wordvec[seq_id], init_state)
            hidden = mx.sym.FullyConnected(data=con, num_hidden=self.embed_size,
                                           weight=input_weight, no_bias=True, name='input_fc')

            if self.use_masking:
                mask = masks[seq_id]

            for i in range(self.layer_num):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = lstm(self.hidden_unit_num, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqid=seq_id, layerid=i, dropout=dp_ratio)
                if self.use_masking:
                    prev_state_h = last_states[i].h
                    prev_state_c = last_states[i].c
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + \
                            mx.sym.broadcast_mul(mask, next_state.h)
                    new_c = mx.sym.broadcast_mul(1.0 - mask, prev_state_c) + \
                            mx.sym.broadcast_mul(mask, next_state.c)
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
        # pylint: disable=invalid-name
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='decoder_softmax', ignore_label=0,
                                  use_ignore=True, normalization='valid')

        return sm
