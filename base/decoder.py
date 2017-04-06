# coding: utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments, import-error
"""A decoder abstract interface object"""
import abc
import six
from network.rnn.lstm import LSTMParam, LSTMState
import mxnet as mx


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
    """ A decoder abstract interface object.

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
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.hidden_unit_num = hidden_unit_num
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.embed_weight = embed_weight
        self.name = name
        self._init_embedding_weight()

    def init_cell_state_parameter(self, init_state):
        """initialize the cell parameter and the initial state(encoder last state)"""
        param_cells = []
        last_states = []
        init_weight = mx.sym.Variable("decoder_init_weight")
        init_bias = mx.sym.Variable("decoder_init_bias")
        # decoder lstm parameters
        init_h = mx.sym.FullyConnected(data=init_state, weight=init_weight, bias=init_bias,
                                       num_hidden=self.hidden_unit_num * self.layer_num,
                                       name='init_fc')
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
        assert len(last_states) == self.layer_num
        return param_cells, last_states

    def _init_embedding_weight(self):
        """word embedding weight"""
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable("{}_embed_weight".format(self.name))

    @staticmethod
    def get_init_state_shape(batch_size, decoder_layer_num, decoder_hidden_unit_num):
        """return init states for lstm decoder"""

        decoder_init_c = [('decoder_l%d_init_c' % l, (batch_size, decoder_hidden_unit_num)) for l in
                          range(decoder_layer_num)]
        decoder_init_states = decoder_init_c

        return decoder_init_states

    @abc.abstractmethod
    def decode(self, init_state):
        """decode process"""
        raise NotImplementedError
