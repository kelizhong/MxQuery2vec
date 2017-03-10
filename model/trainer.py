# -*- coding: utf-8 -*-

import logging

from utils.data_utils import read_data, sentence2id, load_vocab
from symbol import sym_gen
import mxnet as mx
import os
import sys
from masked_bucket_io import MaskedBucketSentenceIter
import time
from utils.file_utils import ensure_dir_exists


class Trainer(object):
    def __init__(self, train_source_path, train_target_path, vocabulary_path, stop_words_dir):
        self.train_source_path = train_source_path
        self.train_target_path = train_target_path
        self.vocabulary_path = vocabulary_path
        self.stop_words_dir = stop_words_dir

    def set_mxnet_parameter(self, **kwargs):
        mxnet_parameter_defaults = {
            "kv_store": "local",
            "monitor_interval": 2,
            "log_level": logging.ERROR,
            "log_path": './logs',
            "save_checkpoint_freq": 100,
            "enable_evaluation": False,
            "invalid_label": 0
        }
        for (parameter, default) in mxnet_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def set_optimizer_parameter(self, **kwargs):
        """optimizer parameter
        Parameter:
            clip_gradient: float, clip gradient in range [-clip_gradient, clip_gradient]
            rescale_grad: float, rescaling factor of gradient. Normally should be 1/batch_size.
        """
        optimizer_defaults = {
            "optimizer": "Adadelta",
            "clip_gradient": 5.0,
            "rescale_grad": -1.0,
        }
        optimizer_params = dict()
        for (parameter, default) in optimizer_defaults.iteritems():
            if parameter == "optimizer":
                # set optimizer name
                setattr(self, parameter, kwargs.get(parameter, default))
            else:
                # set optimizer parameter
                optimizer_params.setdefault(parameter, kwargs.get(parameter, default))
        setattr(self, 'optimizer_params', optimizer_params)
        return self

    def set_model_parameter(self, **kwargs):
        model_parameter_defaults = {
            "source_layer_num": 1,
            "source_hidden_unit_num": 512,
            "source_embed_size": 512,
            "target_layer_num": 1,
            "target_hidden_unit_num": 512,
            "target_embed_size": 512,
            "buckets": [(3, 10), (3, 20), (5, 20), (7, 30)]
        }
        for (parameter, default) in model_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def set_train_parameter(self, **kwargs):
        train_parameter_defaults = {
            "source_dropout": 0.5,
            "target_dropout": 0.5,
            "load_epoch": 0,
            "model_prefix": "query2vec",
            "device_mode": "cpu",
            "devices": 0,
            "train_max_samples": sys.maxsize,
            "disp_batches": 10,
            "batch_size": 128,
            "num_epoch": 65535,
            "num_examples": 65535
        }
        for (parameter, default) in train_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def _load_model(self, rank=0):
        if self.load_epoch is None:
            return (None, None, None)
        assert self.model_prefix is not None
        model_prefix = self.model_prefix
        if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
            model_prefix += "-%d" % (rank)
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, self.load_epoch)
        logging.info('Loaded model %s_%04d.params', model_prefix, self.load_epoch)
        return sym, arg_params, aux_params

    def _save_model(self, rank=0, period=10):
        if self.model_prefix is None:
            return None
        ensure_dir_exists(self.model_prefix, dir_type=False)
        return mx.callback.do_checkpoint(self.model_prefix if rank == 0 else "%s-%d" % (
            self.model_prefix, rank), period)

    def _initial_log(self, kv, log_level, log_path):
        assert kv is not None
        logging.basicConfig(format='Node[' + str(kv.rank) + '] %(asctime)s %(levelname)s:%(name)s:%(message)s',
                            level=log_level,
                            datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
        logging.root.addHandler(file_handler)
        head = 'Node[' + str(kv.rank) + '] %(asctime)s %(levelname)s:%(name)s:%(message)s'
        if log_level is not None and log_path is not None:
            ensure_dir_exists(log_path)
            logging.basicConfig(format=head,
                                level=log_level,
                                datefmt='%H:%M:%S')
            file_handler = logging.FileHandler(os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
            file_handler.setFormatter(logging.Formatter(head))
            logging.root.addHandler(file_handler)
        else:
            logging.basicConfig(level=log_level, format=head)

    def get_LSTM_shape(self):
        # initalize states for LSTM

        forward_source_init_c = [('forward_source_l%d_init_c' % l, (self.batch_size, self.source_hidden_unit_num)) for l
                                 in
                                 range(self.source_layer_num)]
        forward_source_init_h = [('forward_source_l%d_init_h' % l, (self.batch_size, self.source_hidden_unit_num)) for l
                                 in
                                 range(self.source_layer_num)]
        backward_source_init_c = [('backward_source_l%d_init_c' % l, (self.batch_size, self.source_hidden_unit_num)) for
                                  l in
                                  range(self.source_layer_num)]
        backward_source_init_h = [('backward_source_l%d_init_h' % l, (self.batch_size, self.source_hidden_unit_num)) for
                                  l in
                                  range(self.source_layer_num)]
        source_init_states = forward_source_init_c + forward_source_init_h + backward_source_init_c + backward_source_init_h

        target_init_c = [('target_l%d_init_c' % l, (self.batch_size, self.target_hidden_unit_num)) for l in
                         range(self.target_layer_num)]
        target_init_states = target_init_c
        return source_init_states, target_init_states

    def print_all_variable(self):
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))

    def train(self):
        """prepare the data and train"""
        # kvstore
        kv_store = mx.kvstore.create(self.kv_store)
        self._initial_log(kv_store, self.log_level, self.log_path)

        if self.optimizer_params.get('rescale_grad') < 0:
            # if rescale_grad has not been set, reset rescale_grad
            self.optimizer_params.setdefault('rescale_grad', 1.0 / (self.batch_size * kv_store.num_workers))

        # load vocabulary
        vocab = load_vocab(self.vocabulary_path)
        vocab_size = len(vocab) + 1
        logging.info('vocab size: {0}'.format(vocab_size))

        # get states shapes
        source_init_states, target_init_states = self.get_LSTM_shape()

        # build data iterator
        data_loader = MaskedBucketSentenceIter(self.train_source_path, self.train_target_path, self.stop_words_dir,
                                               vocab,
                                               vocab,
                                               self.buckets, self.batch_size,
                                               source_init_states, target_init_states,
                                               text2id=sentence2id, read_data=read_data,
                                               max_read_sample=self.train_max_samples)
        eval_data_loader = None
        network = sym_gen(source_vocab_size=vocab_size, source_layer_num=self.source_layer_num,
                          source_hidden_unit_num=self.source_hidden_unit_num, source_embed_size=self.source_embed_size,
                          source_dropout=self.source_dropout,
                          target_vocab_size=vocab_size, target_layer_num=self.target_layer_num,
                          target_hidden_unit_num=self.target_hidden_unit_num, target_embed_size=self.target_embed_size,
                          target_dropout=self.target_dropout,
                          data_names=data_loader.get_data_names(), label_names=data_loader.get_label_names())

        self._fit(network, kv_store, data_loader, eval_data_loader)

    def _fit(self, network, kv_store, data_loader, eval_data_loader):
        """Train the module parameters
        Args:
            network : the symbol definition of the neural network
            kv_store
            data_loader : function that returns the train data iterators
            eval_data_loader: function that returns the test data iterators
        """

        # load model
        sym, arg_params, aux_params = self._load_model(kv_store.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

        # save model
        checkpoint = self._save_model(kv_store.rank, self.save_checkpoint_freq)

        # devices for training
        if self.device_mode is None or self.device_mode == 'cpu':
            devs = mx.cpu() if self.devices is None or self.devices is '' else [
                mx.cpu(int(i)) for i in self.devices.split(',')]
        else:
            devs = mx.gpu() if self.devices is None or self.devices is '' else [
                mx.gpu(int(i)) for i in self.devices.split(',')]

        # create bucket model
        model = mx.mod.BucketingModule(network, default_bucket_key=data_loader.default_bucket_key, context=devs)

        # set monitor
        monitor = mx.mon.Monitor(self.monitor_interval, pattern=".*") if self.monitor_interval > 0 else None

        # set initializer to initialize the module parameters
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

        # callbacks that run after each batch
        batch_end_callbacks = [mx.callback.Speedometer(self.batch_size, self.disp_batches)]

        # print the variable before training
        if kv_store.rank == 0:
            self.print_all_variable()

        # run
        model.fit(data_loader,
                  begin_epoch=self.load_epoch if self.load_epoch else 0,
                  num_epoch=self.num_epoch,
                  eval_data=eval_data_loader if self.enable_evaluation else None,
                  eval_metric=mx.metric.Perplexity(self.invalid_label),
                  kvstore=kv_store,
                  optimizer=self.optimizer,
                  optimizer_params=self.optimizer_params,
                  initializer=initializer,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  batch_end_callback=batch_end_callbacks,
                  epoch_end_callback=checkpoint,
                  allow_missing=True,
                  monitor=monitor)
