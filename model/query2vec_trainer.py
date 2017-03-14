# -*- coding: utf-8 -*-

import logging
import sys
from itertools import chain

import mxnet as mx

from masked_bucket_io import MaskedBucketSentenceIter
from network.seq2seq.seq2seq_model import encoder_parameter, decoder_parameter, data_label_names_parameter, Seq2seqModel
from trainer import Trainer
from utils.data_util import read_data, sentence2id, load_vocab
from utils.decorator_util import memoized
from utils.device_util import get_devices
from utils.model_util import load_model, save_model, init_log, Speedometer
from utils.tuple_util import namedtuple_with_defaults

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
        end epoch of model training
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
    invalid_label: int
        index for invalid token
    load_epoch: int
        epoch of pretrained model
    train_max_sample: int
        the max sample num for training

"""
mxnet_parameter = namedtuple_with_defaults('mxnet_parameter',
                                           'kv_store hosts_num workers_num device_mode devices num_epoch '
                                           'disp_batches monitor_interval '
                                           'log_level log_path save_checkpoint_freq model_path_prefix '
                                           'enable_evaluation invalid_label load_epoch train_max_samples',
                                           ['local', 1, 1, 'cpu', '0', 65535, 10, 2, logging.ERROR, './logs',
                                            'query2vec', 100,
                                            False, 0, 1, sys.maxsize])

"""optimizer parameter
Parameter:
    optimizer: str
        optimizer method, e.g. Adadelta, sgd
    clip_gradient: float
        clip gradient in range [-clip_gradient, clip_gradient]
    rescale_grad: float
        rescaling factor of gradient. Normally should be 1/batch_size.
"""
optimizer_parameter = namedtuple_with_defaults('optimizer_parameter', 'optimizer clip_gradient rescale_grad',
                                               ['Adadelta', 5.0, -1.0])

"""model parameter
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
model_parameter = namedtuple_with_defaults('model_parameter', 'encoder_layer_num encoder_hidden_unit_num '
                                                              'encoder_embed_size encoder_dropout decoder_layer_num '
                                                              'decoder_hidden_unit_num decoder_embed_size '
                                                              'decoder_dropout batch_size buckets',
                                           [1, 512, 512, 0.0, 1, 512, 512, 0.0, 128,
                                            [(3, 10), (3, 20), (5, 20), (7, 30)]])


class Query2vecTrainer(Trainer):
    def __init__(self, encoder_train_data_path, decoder_train_data_path, vocabulary_path,
                 mxnet_para=mxnet_parameter, optimizer_para=optimizer_parameter,
                 model_para=model_parameter):
        """Trainer for query2vec model
        Args:
            encoder_train_data_path: str
                path for encoder train data
            decoder_train_data_path: str
                path for decoder train data
            vocabulary_path: str
                path for vocabulary
            mxnet_para: namedtuple_with_defaults
                mxnet parameter
            optimizer_para: namedtuple_with_defaults
                optimizer parameter
            model_para: namedtuple_with_defaults
                model parameter
        """
        self.encoder_train_data_path = encoder_train_data_path
        self.decoder_train_data_path = decoder_train_data_path
        self.vocabulary_path = vocabulary_path
        self.mxnet_para = mxnet_para
        self.optimizer_para = optimizer_para
        self.model_para = model_para
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
        setattr(self, 'optimizer_params', optimizer_params)

        if self.optimizer_params.get('rescale_grad') < 0:
            # if rescale_grad has not been set, reset rescale_grad
            self.optimizer_params['rescale_grad'] = 1.0 / (self.batch_size * kv.num_workers)

        # init log with kv
        init_log(self.log_level, self.log_path)

        # print the variable before training
        if kv.rank == 0:
            self.print_all_variable()

    @property
    @memoized
    def init_state_shape(self):
        """initalize states for LSTM"""

        forward_encoder_init_c = [('forward_encoder_l%d_init_c' % l, (self.batch_size, self.encoder_hidden_unit_num))
                                  for l
                                  in
                                  range(self.encoder_layer_num)]
        forward_encoder_init_h = [('forward_encoder_l%d_init_h' % l, (self.batch_size, self.encoder_hidden_unit_num))
                                  for l
                                  in
                                  range(self.encoder_layer_num)]
        backward_encoder_init_c = [('backward_encoder_l%d_init_c' % l, (self.batch_size, self.encoder_hidden_unit_num))
                                   for
                                   l in
                                   range(self.encoder_layer_num)]
        backward_encoder_init_h = [('backward_encoder_l%d_init_h' % l, (self.batch_size, self.encoder_hidden_unit_num))
                                   for
                                   l in
                                   range(self.encoder_layer_num)]
        encoder_init_states = forward_encoder_init_c + forward_encoder_init_h + backward_encoder_init_c + \
                              backward_encoder_init_h

        decoder_init_c = [('decoder_l%d_init_c' % l, (self.batch_size, self.decoder_hidden_unit_num)) for l in
                          range(self.decoder_layer_num)]
        decoder_init_states = decoder_init_c
        return encoder_init_states, decoder_init_states

    @property
    @memoized
    def vocab(self):
        """load vocabulary"""
        vocab = load_vocab(self.vocabulary_path)
        return vocab

    @property
    @memoized
    def vocab_size(self):
        """return vocabulary size"""
        return len(self.vocab)

    @property
    @memoized
    def data_loader(self):
        # get states shapes
        encoder_init_states, decoder_init_states = self.init_state_shape
        # build data iterator
        data_loader = MaskedBucketSentenceIter(self.encoder_train_data_path, self.decoder_train_data_path,
                                               self.vocab,
                                               self.vocab,
                                               self.buckets, self.batch_size,
                                               encoder_init_states, decoder_init_states,
                                               sentence2id=sentence2id, read_data=read_data,
                                               max_read_sample=self.train_max_samples)
        return data_loader

    @property
    def eval_data_loader(self):
        return None

    @property
    def network_symbol(self):

        encoder_para = encoder_parameter(encoder_vocab_size=self.vocab_size, encoder_layer_num=self.encoder_layer_num,
                                         encoder_hidden_unit_num=self.encoder_hidden_unit_num,
                                         encoder_embed_size=self.encoder_embed_size,
                                         encoder_dropout=self.encoder_dropout)

        decoder_para = decoder_parameter(decoder_vocab_size=self.vocab_size, decoder_layer_num=self.decoder_layer_num,
                                         decoder_hidden_unit_num=self.decoder_hidden_unit_num,
                                         decoder_embed_size=self.decoder_embed_size,
                                         decoder_dropout=self.decoder_dropout)

        data_label_names_para = data_label_names_parameter(data_names=self.data_loader.get_data_names(),
                                                           label_names=self.data_loader.get_label_names())

        sym = Seq2seqModel(encoder_para, decoder_para, data_label_names_para).network_symbol()
        return sym

    @memoized
    @property
    def devices(self):
        """return devices"""
        devs = get_devices(self.devices, self.device_mode, self.rank, self.hosts_num, self.workers_num)
        return devs

    def print_all_variable(self):
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))

    def train(self):
        """Train the module parameters"""

        network_symbol = self.network_symbol
        data_loader = self.data_loader
        eval_data_loader = self.eval_data_loader

        # load model
        sym, arg_params, aux_params = load_model(self.model_path_prefix, self.kv.rank, self.load_epoch)
        if sym is not None:
            assert sym.tojson() == network_symbol.tojson()

        # save model
        checkpoint = save_model(self.model_path_prefix, self.kv.rank, self.save_checkpoint_freq)

        devices = self.devices

        # create bucket model
        model = mx.mod.BucketingModule(network_symbol, default_bucket_key=data_loader.default_bucket_key,
                                       context=devices)

        # set monitor
        monitor = mx.mon.Monitor(self.monitor_interval, pattern=".*") if self.monitor_interval > 0 else None

        # set initializer to initialize the module parameters
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

        # callbacks that run after each batch
        batch_end_callbacks = [Speedometer(self.batch_size, self.kv.rank, self.disp_batches)]

        # run
        model.fit(data_loader,
                  begin_epoch=self.load_epoch if self.load_epoch else 0,
                  num_epoch=self.num_epoch,
                  eval_data=eval_data_loader if self.enable_evaluation else None,
                  eval_metric=mx.metric.Perplexity(self.invalid_label),
                  kvstore=self.kv,
                  optimizer=self.optimizer,
                  optimizer_params=self.optimizer_params,
                  initializer=initializer,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  batch_end_callback=batch_end_callbacks,
                  epoch_end_callback=checkpoint,
                  allow_missing=True,
                  monitor=monitor)
