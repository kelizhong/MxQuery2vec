import six
import abc
from collections import namedtuple
from itertools import chain
encoder_parameter = namedtuple('encoder_parameter', 'encoder_layer_num encoder_vocab_size '
                                                    'encoder_hidden_unit_num encoder_embed_size encoder_dropout')

decoder_parameter = namedtuple('decoder_parameter', 'decoder_layer_num decoder_vocab_size '
                                                    'decoder_hidden_unit_num decoder_embed_size decoder_dropout')


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """A seq2seq model abstract interface object."""

    def __init__(self, encoder_para, decoder_para, share_embed=True):
        self.encoder_para = encoder_para
        self.decoder_para = decoder_para
        self.share_embed = share_embed
        self._initialize()

    def _initialize(self):
        assert isinstance(self.encoder_para, encoder_parameter)
        assert isinstance(self.decoder_para, decoder_parameter)

        for (parameter, value) in chain(self.encoder_para._asdict().iteritems(),
                                        self.decoder_para._asdict().iteritems()):
            setattr(self, parameter, value)

    @abc.abstractmethod
    def encoder(self, seq_len):
        """encoder graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def decoder(self, seq_len):
        """decoder graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def attention(self, seq_len):
        """attention mechanism"""
        raise NotImplementedError

    @abc.abstractmethod
    def graph(self, encoder_seq_len, decoder_seq_len):
        """build the network graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def network_symbol(self):
        """build the network symbol for mxnet training using network graph"""
        raise NotImplementedError

    @property
    def embed_weight(self):
        """return word embedding weight"""
        raise NotImplementedError
