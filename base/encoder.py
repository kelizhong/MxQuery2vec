# coding: utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments, import-error
"""encoder abstract interface object"""
import abc
import six
from network.rnn.lstm import LSTMParam, LSTMState
import mxnet as mx


@six.add_metaclass(abc.ABCMeta)
class Encoder(object):
    """ A encoder abstract interface object.
        X = Input sequence
        C = LSTM(X); The context vector
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
            encoder name
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
        self._init_embedding_weight()

    def init_cell_parameter(self):
        """encoder bi-lstm parameters"""
        forward_param_cells = []
        forward_last_states = []
        backward_param_cells = []
        backward_last_states = []
        # forward part
        for i in range(self.layer_num):
            # pylint: disable=line-too-long
            forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("forward_encoder_l%d_i2h_weight" % i),
                                                 i2h_bias=mx.sym.Variable("forward_encoder_l%d_i2h_bias" % i),
                                                 h2h_weight=mx.sym.Variable("forward_encoder_l%d_h2h_weight" % i),
                                                 h2h_bias=mx.sym.Variable("forward_encoder_l%d_h2h_bias" % i)))
            forward_state = LSTMState(c=mx.sym.Variable("forward_encoder_l%d_init_c" % i),
                                      h=mx.sym.Variable("forward_encoder_l%d_init_h" % i))
            forward_last_states.append(forward_state)
        assert len(forward_last_states) == self.layer_num, \
            "shape not match between forward_last_states and layer_num for encoder"
        # backward part
        for i in range(self.layer_num):
            backward_param_cells.append(
                LSTMParam(i2h_weight=mx.sym.Variable("backward_encoder_l%d_i2h_weight" % i),
                          i2h_bias=mx.sym.Variable("backward_encoder_l%d_i2h_bias" % i),
                          h2h_weight=mx.sym.Variable("backward_encoder_l%d_h2h_weight" % i),
                          h2h_bias=mx.sym.Variable("backward_encoder_l%d_h2h_bias" % i)))
            backward_state = LSTMState(c=mx.sym.Variable("backward_encoder_l%d_init_c" % i),
                                       h=mx.sym.Variable("backward_encoder_l%d_init_h" % i))
            backward_last_states.append(backward_state)
        assert len(backward_last_states) == self.layer_num, \
            "shape not match between backward_last_states and layer_num for encoder"
        return forward_param_cells, forward_last_states, backward_param_cells, backward_last_states

    def _init_embedding_weight(self):
        """word embedding weight"""
        if self.embed_weight is None:
            self.embed_weight = mx.sym.Variable("{}_embed_weight".format(self.name))

    @staticmethod
    def get_encoder_last_state(forward_hidden_state, backward_hidden_state):
        """return the last state. Decoder use it to initialize the state"""
        encoder_last_state = mx.sym.Concat(forward_hidden_state, backward_hidden_state, dim=1,
                                           name='encoder_last_state')

        return encoder_last_state

    @staticmethod
    def get_init_state_shape(batch_size, encoder_layer_num, encoder_hidden_unit_num):
        """return init-states for bi-LSTM"""

        # pylint: disable=line-too-long
        forward_encoder_init_c = [('forward_encoder_l%d_init_c' % l, (batch_size, encoder_hidden_unit_num))
                                  for l in range(encoder_layer_num)]
        forward_encoder_init_h = [('forward_encoder_l%d_init_h' % l, (batch_size, encoder_hidden_unit_num))
                                  for l in range(encoder_layer_num)]
        backward_encoder_init_c = [('backward_encoder_l%d_init_c' % l, (batch_size, encoder_hidden_unit_num))
                                   for l in range(encoder_layer_num)]
        backward_encoder_init_h = [('backward_encoder_l%d_init_h' % l, (batch_size, encoder_hidden_unit_num))
                                   for l in range(encoder_layer_num)]
        encoder_init_states = forward_encoder_init_c + forward_encoder_init_h + backward_encoder_init_c + \
                              backward_encoder_init_h

        return encoder_init_states

    @abc.abstractmethod
    def encode(self):
        """encode process"""
        raise NotImplementedError
