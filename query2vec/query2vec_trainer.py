# -*- coding: utf-8 -*-

import logging
import sys
import mxnet as mx
import numpy as np
from base.trainer import Trainer
from masked_bucket_io import MaskedBucketSentenceIter
from metric.seq2seq_metric import MetricManage
from metric.speedometer import Speedometer
from network.seq2seq.seq2seq_model import encoder_parameter, decoder_parameter, Seq2seqModel
from utils.data_util import read_data, sentence2id, load_pickle_object
from utils.decorator_util import memoized
from utils.model_util import load_model, save_model
from utils.record_type import RecordType

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
    ignore_label: int
        index for ignore_label token
    load_epoch: int
        epoch of pretrained model
    train_max_sample: int
        the max sample num for training

"""

mxnet_parameter = RecordType('mxnet_parameter', [('kv_store', 'local'), ('hosts_num', 1), ('workers_num', 1),
                                                 ('device_mode', 'cpu'), ('devices', '0'), ('num_epoch', 65535),
                                                 ('disp_batches', 1), ('monitor_interval', -1),
                                                 ('log_level', logging.ERROR), ('log_path', './logs'),
                                                 ('save_checkpoint_freq', 1),
                                                 ('model_path_prefix', 'query2vec'), ('enable_evaluation', False),
                                                 ('ignore_label', 0),
                                                 ('load_epoch', -1), ('train_max_samples', sys.maxsize),
                                                 ('word2vec_path', './data/word2vec/model/w2v.pkl')])
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
optimizer_parameter = RecordType('optimizer_parameter',
                                 [('optimizer', 'Adadelta'), ('clip_gradient', 5.0), ('rescale_grad', -1.0),
                                  ('learning_rate', 0.01), ('wd', 0.0005), ('momentum', 0.9)])

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

model_parameter = RecordType('model_parameter',
                             [('encoder_layer_num', 1), ('encoder_hidden_unit_num', 256), ('encoder_embed_size', 128),
                              ('encoder_dropout', 0.0), ('decoder_layer_num', 1), ('decoder_hidden_unit_num', 256),
                              ('decoder_embed_size', 128), ('decoder_dropout', 0.0), ('batch_size', 128),
                              ('buckets', [(3, 10), (3, 20), (5, 20), (7, 30)])])


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
            mxnet_para: RecordType
                mxnet parameter
            optimizer_para: RecordType
                optimizer parameter
            model_para: RecordType
                model parameter
        """
        super(Query2vecTrainer, self).__init__(mxnet_para=mxnet_para, optimizer_para=optimizer_para,
                                               model_para=model_para)
        self.encoder_train_data_path = encoder_train_data_path
        self.decoder_train_data_path = decoder_train_data_path
        self.vocabulary_path = vocabulary_path

    @property
    @memoized
    def vocab(self):
        """load vocabulary"""
        vocab = load_pickle_object(self.vocabulary_path)
        return vocab

    @property
    @memoized
    def word2vec(self):
        """load pretrain word2vec"""
        if not self.word2vec_path:
            print("Fsdf")
        w2v = load_pickle_object(self.word2vec_path) if not self.word2vec_path else None
        return w2v

    @property
    @memoized
    def vocab_size(self):
        """return vocabulary size"""
        return len(self.vocab) + 1

    @property
    @memoized
    def network_symbol(self):

        encoder_para = encoder_parameter(encoder_vocab_size=self.vocab_size, encoder_layer_num=self.encoder_layer_num,
                                         encoder_hidden_unit_num=self.encoder_hidden_unit_num,
                                         encoder_embed_size=self.encoder_embed_size,
                                         encoder_dropout=self.encoder_dropout)

        decoder_para = decoder_parameter(decoder_vocab_size=self.vocab_size, decoder_layer_num=self.decoder_layer_num,
                                         decoder_hidden_unit_num=self.decoder_hidden_unit_num,
                                         decoder_embed_size=self.decoder_embed_size,
                                         decoder_dropout=self.decoder_dropout)

        sym = Seq2seqModel(encoder_para, decoder_para)
        return sym

    @property
    @memoized
    def train_data_loader(self):
        # get states shapes
        encoder_init_states, decoder_init_states = self.network_symbol.get_bi_init_state_shape(self.batch_size)
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

    def _load_model_with_pretrain_word2vec(self, embed_weight_name):
        sym, arg_params, aux_params = load_model(self.model_path_prefix, self.kv.rank, self.load_epoch)
        # load the pretrain word2vec only for new model training and word2vec_path is defined
        if not arg_params or not self.word2vec_path:
            return sym, arg_params, aux_params
        w2v = self.word2vec
        vocab = self.vocab
        embed_weight = np.random.rand(self.vocab_size, self.encoder_embed_size)

        for word, id in vocab.iteritems():
            if word in w2v:
                embed_weight[id] = w2v[word]
        arg_params = dict()
        arg_params[embed_weight_name] = mx.nd.array(embed_weight)
        return sym, arg_params, aux_params

    def train(self):
        """Train the module parameters"""

        train_data_loader = self.train_data_loader
        eval_data_loader = self.eval_data_loader
        network_symbol = self.network_symbol.network_symbol(train_data_loader.data_names, train_data_loader.label_names)

        # load model
        sym, arg_params, aux_params = self._load_model_with_pretrain_word2vec('share_embed_weight')

        if sym is not None:
            assert sym.tojson() == network_symbol.tojson()

        # save model
        checkpoint = save_model(self.model_path_prefix, self.kv.rank, self.save_checkpoint_freq)

        devices = self.ctx_devices
        # create bucket model
        model = mx.mod.BucketingModule(network_symbol, default_bucket_key=train_data_loader.default_bucket_key,
                                       context=devices)

        # set monitor
        monitor = mx.mon.Monitor(self.monitor_interval, pattern=".*") if self.monitor_interval > 0 else None

        # set initializer to initialize the module parameters
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

        # callbacks that run after each batch
        batch_end_callbacks = [Speedometer(self.batch_size, self.kv.rank, self.disp_batches)]

        metric_manager = MetricManage(self.ignore_label)
        metrics = [metric_manager.create_metric('perplexity')]
        # run
        model.fit(train_data_loader,
                  begin_epoch=self.load_epoch if self.load_epoch else 0,
                  num_epoch=self.num_epoch,
                  eval_data=eval_data_loader if self.enable_evaluation else None,
                  eval_metric=metrics,
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
