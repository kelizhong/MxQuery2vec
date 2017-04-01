import six
import abc
from utils.device_util import get_devices
from utils.decorator_util import memoized
import logging
from utils.record_util import RecordType
from itertools import chain
import mxnet as mx


@six.add_metaclass(abc.ABCMeta)
class Trainer(object):
    """A trainer abstract interface object."""

    def __init__(self, mxnet_para, optimizer_para, model_para, data_para):
        self._mxnet_para = mxnet_para
        self._optimizer_para = optimizer_para
        self._model_para = model_para
        self._data_para = data_para
        self._init_parameter()
        # print the variable before training
        if self.kv.rank == 0:
            self.print_all_variable()

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
    def train_data_loader(self):
        """return train data loader"""
        raise NotImplementedError

    @property
    def eval_data_loader(self):
        """return evaluation data loader"""
        raise NotImplementedError

    @property
    def model(self):
        """return model instance"""
        raise NotImplementedError

    def print_all_variable(self):
        """log training parameter to check before training"""
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))

    def _init_parameter(self):
        """initialize mxnet, optimizer, model, data parameter and kv store for training"""
        assert isinstance(self._mxnet_para, RecordType)
        assert isinstance(self._optimizer_para, RecordType)
        assert isinstance(self._model_para, RecordType)
        assert isinstance(self._data_para, RecordType)

        for (parameter, value) in chain(self._mxnet_para.iteritems(),
                                        self._model_para.iteritems(),
                                        self._data_para.iteritems()):
            setattr(self, parameter, value)

        # create kvstore
        kv = mx.kvstore.create(self.kv_store)
        setattr(self, 'kv', kv)

        optimizer_params = dict()
        for (parameter, value) in self._optimizer_para.iteritems():
            if parameter == "optimizer":
                # set optimizer name
                setattr(self, parameter, value)
            else:
                # set optimizer parameter
                optimizer_params.setdefault(parameter, value)
        if self.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
            optimizer_params.pop('momentum')
        setattr(self, 'optimizer_params', optimizer_params)

        if self.optimizer_params.get('rescale_grad') < 0:
            # if rescale_grad has not been set, reset rescale_grad
            self.optimizer_params['rescale_grad'] = 1.0 / (self.batch_size * kv.num_workers)

