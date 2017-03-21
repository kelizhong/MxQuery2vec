import six
import abc


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """A seq2seq model abstract interface object."""

    @abc.abstractmethod
    def encoder(self, seq_len):
        raise NotImplementedError

    @abc.abstractmethod
    def decoder(self, seq_len):
        raise NotImplementedError

    @abc.abstractmethod
    def attention(self, seq_len):
        raise NotImplementedError

    @abc.abstractmethod
    def unroll(self, encoder_seq_len, decoder_seq_len):
        raise NotImplementedError

    @abc.abstractmethod
    def network_symbol(self):
        raise NotImplementedError

    @property
    def embed_weight(self):
        """return word embedding weight"""
        raise NotImplementedError
