import six
import abc


@six.add_metaclass(abc.ABCMeta)
class Trainer(object):
    """A trainer abstract interface object."""

    @abc.abstractmethod
    def train(self):
        """train process"""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_devices(self):
        """return devices"""
        raise NotImplementedError

    @property
    def init_state_shape(self):
        """initalize states for network"""
        raise NotImplementedError

    @property
    def vocab(self):
        """return vocabulary"""
        raise NotImplementedError

    @property
    def vocab_size(self):
        """return vocabulary size"""
        raise NotImplementedError

    @property
    def data_loader(self):
        raise NotImplementedError

    @property
    def eval_data_loader(self):
        raise NotImplementedError

    @property
    def network_symbol(self):
        raise NotImplementedError


