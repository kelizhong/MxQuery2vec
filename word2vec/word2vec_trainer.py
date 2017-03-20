from wordvec_model import Word2vec
import mxnet as mx
from utils.decorator_util import memoized
from utils.device_util import get_devices
from metric.word2vec_metric import NceAuc
from metric.speedometer import Speedometer
from word2vec_io import Word2vecDataIter
from utils.model_util import load_model, save_model, init_log
from utils.tuple_util import namedtuple_with_defaults
import logging
from itertools import chain
from base.trainer import Trainer

"""mxnet parameter
Parameter:
    kv_store: str, the type of KVStore
        - local works for multiple devices on a single machine (single process)
        - dist works for multi-machines (multiple processes)
    hosts_num: int
        the number of hosts
    workers_num: int
        the number of workers
    device_mode: str
        device mode, ['cpu', 'gpu', 'gpu_auto']
    devices: str
        the devices will be used, e.g "0,1,2,3"
    num_epoch: int
        end epoch of query2vec training
    disp_batches: int
        show progress for every n batches
    monitor_interval: int
        Number of batches between printing.
    log_level: log level
    log_path: str
        path to store log
    save_checkpoint_freq: int
        the frequency to save checkpoint
    enable_evaluation: boolean
        whether to enable evaluation
    ignore_label: int
        index for ignore_label token
    load_epoch: int
        epoch of pretrained query2vec
    train_max_sample: int
        the max sample num for training

"""
mxnet_parameter = namedtuple_with_defaults('mxnet_parameter',
                                           'kv_store hosts_num workers_num device_mode devices num_epoch '
                                           'disp_batches monitor_interval '
                                           'log_level log_path save_checkpoint_freq model_path_prefix '
                                           'enable_evaluation load_epoch',
                                           ['local', 1, 1, 'cpu', '0', 65535, 10, 2, logging.ERROR, './logs',
                                            'word2vec', 100,
                                            False,  1])

"""optimizer parameter
Parameter:
    optimizer: str
        optimizer method, e.g. Adadelta, sgd
    clip_gradient: float
        clip gradient in range [-clip_gradient, clip_gradient]
    rescale_grad: float
        rescaling factor of gradient. Normally should be 1/batch_size.
    learning_rate: float
        learning rate of the stochastic gradient descent
    momentum: float
        momentum for sgd
    wd: float
        weight decay
"""
optimizer_parameter = namedtuple_with_defaults('optimizer_parameter',
                                               'optimizer clip_gradient rescale_grad learning_rate wd momentum',
                                               ['Adadelta', 5.0, -1.0, 0.01, 0.0005, 0.9])


"""query2vec parameter
Parameter:
    encoder_layer_num: int
        number of layers for the LSTM recurrent neural network for encoder
    encoder_hidden_unit_num: int
        number of hidden units in the neural network for encoder
    encoder_embed_size: int
        word embedding size for encoder
    encoder_dropout: float
        the probability to ignore the neuron outputs
    decoder_layer_num: int
        number of layers for the LSTM recurrent neural network for decoder
    decoder_hidden_unit_num: int
        number of hidden units in the neural network for decoder
    decoder_embed_size: int
        word embedding size for decoder
    decoder_dropout: float
        the probability to ignore the neuron outputs
    batch_size: int
        batch size for each databatch'
    buckets: tuple list
        bucket for encoder sequence length and decoder sequence length
"""
model_parameter = namedtuple_with_defaults('model_parameter', 'embed_size batch_size window_size',
                                           [128, 128, 2])


class Word2vecTrainer(Trainer):
    def __init__(self, data_path, vocabulary_save_path, mxnet_para=mxnet_parameter, optimizer_para=optimizer_parameter,
                 model_para=model_parameter):
        self.mxnet_para = mxnet_para
        self.optimizer_para = optimizer_para
        self.model_para = model_para
        self.data_path = data_path #./data/word2vec/train.dec
        self.vocabulary_save_path = vocabulary_save_path
        self._initialize()

    def _initialize(self):
        assert isinstance(self.mxnet_para, mxnet_parameter)
        assert isinstance(self.optimizer_para, optimizer_parameter)
        assert isinstance(self.model_para, model_parameter)

        for (parameter, value) in chain(self.mxnet_para._asdict().iteritems(),
                                        self.model_para._asdict().iteritems()):
            setattr(self, parameter, value)

        # create kvstore
        kv = mx.kvstore.create(self.kv_store)
        setattr(self, 'kv', kv)

        optimizer_params = dict()
        for (parameter, value) in self.optimizer_para._asdict().iteritems():
            if parameter == "optimizer":
                # set optimizer name
                setattr(self, parameter, value)
            else:
                # set optimizer parameter
                optimizer_params.setdefault(parameter, value)

        if self.optimizer.lower() in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
            optimizer_params.__delitem__('momentum')
        setattr(self, 'optimizer_params', optimizer_params)

        if self.optimizer_params.get('rescale_grad') < 0:
            # if rescale_grad has not been set, reset rescale_grad
            self.optimizer_params['rescale_grad'] = 1.0 / (self.batch_size * kv.num_workers)

        # init log with kv
        init_log(self.log_level, self.log_path)

        # print the variable before training
        if kv.rank == 0:
            self.print_all_variable()

    def print_all_variable(self):
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))

    @property
    @memoized
    def ctx_devices(self):
        """return devices"""
        devs = get_devices(self.devices, self.device_mode, self.kv.rank, self.hosts_num, self.workers_num)
        return devs

    @property
    def network_symbol(self):
        sym = Word2vec(self.batch_size, self.data_loader.vocab_size, self.embed_size, self.window_size).network_symbol()
        return sym

    @property
    @memoized
    def data_loader(self):
        # build data iterator
        data_loader = Word2vecDataIter(self.data_path, self.vocabulary_save_path, self.batch_size, 2 * self.window_size + 1)
        return data_loader

    @property
    def eval_data_loader(self):
        return None

    def train(self):
        network_symbol = self.network_symbol
        devices = self.ctx_devices
        data_loader = self.data_loader

        # load query2vec
        sym, arg_params, aux_params = load_model(self.model_path_prefix, self.kv.rank, self.load_epoch)
        # save query2vec
        checkpoint = save_model(self.model_path_prefix, self.kv.rank, self.save_checkpoint_freq)

        # set initializer to initialize the module parameters
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)
        model = mx.mod.Module(context=devices,
                              symbol=network_symbol, data_names=self.data_loader.data_names,
                              label_names=self.data_loader.label_names)

        # callbacks that run after each batch
        batch_end_callbacks = [Speedometer(self.batch_size, self.kv.rank, self.disp_batches)]

        metric = NceAuc()
        model.fit(data_loader,
                  eval_metric=metric,
                  initializer=initializer,
                  num_epoch=self.num_epoch,
                  batch_end_callback=batch_end_callbacks,
                  epoch_end_callback=checkpoint,
                  kvstore=self.kv,
                  optimizer=self.optimizer,
                  optimizer_params=self.optimizer_params,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  )
