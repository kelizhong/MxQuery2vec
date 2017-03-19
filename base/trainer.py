import six
import abc
from utils.device_util import get_devices
from utils.decorator_util import memoized
import logging


@six.add_metaclass(abc.ABCMeta)
class Trainer(object):
    """A trainer abstract interface object."""

    @abc.abstractmethod
    def train(self):
        """train process"""
        raise NotImplementedError

    @property
    @memoized
    def ctx_devices(self):
        """return devices"""
        devs = get_devices(self.devices, self.device_mode, self.kv.rank, self.hosts_num, self.workers_num)
        return devs

    @property
    def data_loader(self):
        raise NotImplementedError

    @property
    def eval_data_loader(self):
        raise NotImplementedError

    @property
    def network_symbol(self):
        raise NotImplementedError

    def print_all_variable(self):
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))


