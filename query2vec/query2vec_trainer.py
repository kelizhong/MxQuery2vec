# coding=utf-8
# pylint: disable=no-member, invalid-name. import-error, pointless-string-statement
"""query2vec trainer implementation"""
import logbook as logging
import sys

import mxnet as mx
import numpy as np

from base.model import encoder_parameter, decoder_parameter
from base.trainer import Trainer
from data_io.distribute_stream.aksis_data_receiver import AksisDataReceiver
from data_io.data_stream.seq2seq_data_stream import Seq2seqDataStream
from data_io.seq2seq_bucket_io_iter import Seq2seqMaskedBucketIoIter
from metric.seq2seq_metric import MetricManage
from metric.speedometer import Speedometer
from network.seq2seq.seq2seq_model import Seq2seqModel
from utils.data_util import load_vocabulary_from_pickle
from utils.pickle_util import load_pickle_object
from utils.decorator_util import memoized
from utils.model_util import load_model, save_model_callback
from utils.record_util import RecordType
from common.constant import special_words

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
        number of batches between printing.
    save_checkpoint_freq: int
        the frequency to save checkpoint
    model_path_prefix: str
        the prefix for the parameters file
    enable_evaluation: boolean
        whether to enable evaluation
    ignore_label: int
        index for ignore_label token
    load_epoch: int
        epoch of pretrained model
    train_max_sample: int
        the max sample num for training, only use it for file data stream
    monitor_pattern: str
        a regular expression specifying which tensors to monitor.
        Only tensors with names that match name_pattern will be included.
        For example, '.*weight|.*output' will print all weights and outputs;
        '.*backward.*' will print all gradients.
    metric: str
        the performance measure which defined in seq2seq_metric.py used to display during training.
    save_checkpoint_x_batch: int
        save checkpoint every x batch, only available in zmq data stream

"""

mxnet_parameter = RecordType('mxnet_parameter', [('kv_store', 'local'), ('hosts_num', 1), ('workers_num', 1),
                                                 ('device_mode', 'cpu'), ('devices', '0'), ('num_epoch', 65535),
                                                 ('disp_batches', 1), ('monitor_interval', -1),
                                                 ('save_checkpoint_freq', 1),
                                                 ('model_path_prefix', './data/query2vec/model/query2vec'),
                                                 ('enable_evaluation', False),
                                                 ('ignore_label', 0),
                                                 ('load_epoch', -1), ('train_max_samples', sys.maxsize),
                                                 ('monitor_pattern', '.*'), ('metric', 'perplexity'),
                                                 ('save_checkpoint_x_batch', 1000)])

"""data parameter
Parameter:
    encoder_train_data_path: str
        encoder corpus data path, only for file data stream
    decoder_train_data_path: str
        decoder corpus data path, only for file data stream
    vocabulary_path: str
        vocabulary path for corpus
    top_words: int
        the max words num for training
    word2vec_path: str
        path for word2vec model, which is used to initialize the embedding layer instead of random
    ip_addr: str
        ip address to receiver the train data
    port: str
        ip port to receiver the train data


"""
data_parameter = RecordType('data_parameter',
                            [('encoder_train_data_path', None), ('decoder_train_data_path', None),
                             ('vocabulary_path', ''), ('top_words', 40000),
                             ('word2vec_path', None), ('ip_addr', None), ('port', None)])
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
        batch size for each databatch
    buckets: tuple list
        bucket for encoder sequence length and decoder sequence length
"""

model_parameter = RecordType('model_parameter',
                             [('encoder_layer_num', 1), ('encoder_hidden_unit_num', 256), ('encoder_embed_size', 128),
                              ('encoder_dropout', 0.0), ('decoder_layer_num', 1), ('decoder_hidden_unit_num', 256),
                              ('decoder_embed_size', 128), ('decoder_dropout', 0.0), ('batch_size', 128),
                              ('buckets', [(3, 10), (3, 20), (5, 20), (7, 30)])])


class Query2vecTrainer(Trainer):
    """Trainer for query2vec model
    Parameter:
        mxnet_para: RecordType
            mxnet parameter
        optimizer_para: RecordType
            optimizer parameter
        model_para: RecordType
            model parameter
        data_para: RecordType
            data parameter
    """

    def __init__(self,
                 mxnet_para=mxnet_parameter, optimizer_para=optimizer_parameter,
                 model_para=model_parameter, data_para=data_parameter):
        super(Query2vecTrainer, self).__init__(mxnet_para=mxnet_para, optimizer_para=optimizer_para,
                                               model_para=model_para, data_para=data_para)
        self.check_args()

    def check_args(self):
        """validate argument"""
        if (self.ip_addr or self.port) and (self.encoder_train_data_path or self.decoder_train_data_path):
            logging.error(
                "ip_addr and port|encoder_train_data_path and decoder_train_data_path are mutually exclusive ...")
            sys.exit(2)

    @property
    @memoized
    def vocab(self):
        """load vocabulary"""
        vocab = load_vocabulary_from_pickle(self.vocabulary_path, top_words=self.top_words, special_words=special_words)
        return vocab

    @property
    @memoized
    def word2vec(self):
        """load pretrain word2vec"""
        w2v = load_pickle_object(self.word2vec_path) if self.word2vec_path else None
        return w2v

    @property
    @memoized
    def vocab_size(self):
        """return vocabulary size"""
        return len(self.vocab)

    @property
    @memoized
    def model(self):
        """create seq2seq model"""
        encoder_para = encoder_parameter(encoder_vocab_size=self.vocab_size, encoder_layer_num=self.encoder_layer_num,
                                         encoder_hidden_unit_num=self.encoder_hidden_unit_num,
                                         encoder_embed_size=self.encoder_embed_size,
                                         encoder_dropout=self.encoder_dropout)

        decoder_para = decoder_parameter(decoder_vocab_size=self.vocab_size, decoder_layer_num=self.decoder_layer_num,
                                         decoder_hidden_unit_num=self.decoder_hidden_unit_num,
                                         decoder_embed_size=self.decoder_embed_size,
                                         decoder_dropout=self.decoder_dropout)

        m = Seq2seqModel(encoder_para, decoder_para)
        return m

    @property
    def data_stream_from_file(self):
        """file data stream which accept the encoder and decoder data path, this data stream is mainly used for test"""
        data_stream = Seq2seqDataStream(self.encoder_train_data_path, self.decoder_train_data_path, self.vocab,
                                        self.vocab, self.buckets, self.batch_size,
                                        max_sentence_num=self.train_max_samples)
        return data_stream

    @property
    def data_stream_from_zmq(self):
        """listen zmq socket to receiver train data"""
        data_stream = AksisDataReceiver(self.ip_addr, self.port, self.save_checkpoint_x_batch)
        return data_stream

    @property
    def data_stream(self):
        """return file data stream if encoder_train_data_path and decoder_train_data_path are defined.
           return zmq data stream if ip_addr and port are defined.
           else raise RuntimeError
        """
        if self.encoder_train_data_path and self.decoder_train_data_path:
            data_stream = self.data_stream_from_file
        elif self.ip_addr and self.port:
            data_stream = self.data_stream_from_zmq
        else:
            raise RuntimeError("fail to get data stream")
        return data_stream

    @property
    @memoized
    def train_data_loader(self):
        """Mxnet require an IO iter to load the train data, so feed the data from data stream into mxnet io iter"""
        # get states shapes
        encoder_init_states, decoder_init_states = self.model.get_init_state_shape(self.batch_size)
        # build data iterator
        # data_stream = Seq2seqDataStream(self.encoder_train_data_path, self.decoder_train_data_path, self.vocab,
        #                                self.vocab, self.buckets, self.batch_size,
        #                                max_sentence_num=self.train_max_samples)
        data_stream = self.data_stream
        data_loader = Seq2seqMaskedBucketIoIter(data_stream,
                                                encoder_init_states, decoder_init_states, max(self.buckets),
                                                self.batch_size)

        return data_loader

    @property
    def eval_data_loader(self):
        # TODO implement evaluation data loader
        return None

    def _load_model_with_pretrain_word2vec(self, embed_weight_name):
        """initialize the embedding layer parameter with the pretrain word2vec if word2vec exists"""
        rank = self.kv.rank
        sym, arg_params, aux_params = load_model(self.model_path_prefix, rank, self.load_epoch)
        # load the pretrain word2vec only for new model training and word2vec_path is defined
        if arg_params is not None or self.word2vec_path is None:
            logging.info("Worker {} create model without pretrain word2vec model", rank)
            return sym, arg_params, aux_params
        logging.info("Worker {} initialize embedding weight using pretrain word2vec model", rank)
        w2v = self.word2vec
        vocab = self.vocab
        embed_weight = np.random.rand(self.vocab_size, self.encoder_embed_size)

        for word, index in vocab.iteritems():
            if word in w2v:
                embed_weight[index] = w2v[word]
        arg_params = dict()
        arg_params[embed_weight_name] = mx.nd.array(embed_weight)
        return sym, arg_params, aux_params

    def train(self):
        """Train the module parameters"""
        rank = self.kv.rank
        # load model
        logging.info("Worker {} loading model", rank)
        sym, arg_params, aux_params = self._load_model_with_pretrain_word2vec('share_embed_weight')

        # save model callback
        checkpoint = save_model_callback(self.model_path_prefix, rank, self.save_checkpoint_freq)

        devices = self.ctx_devices
        logging.info("Worker {} using devices {}", rank, devices)

        # set monitor
        monitor = mx.mon.Monitor(self.monitor_interval,
                                 pattern=self.monitor_pattern) if self.monitor_interval > 0 and \
                                                                  self.monitor_pattern is not None else None

        # set initializer to initialize the module parameters
        # np plan to tune this initializer parameter, just hard code here
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

        # callbacks that run after each batch
        batch_end_callbacks = [Speedometer(self.batch_size, self.kv.rank, self.disp_batches)]

        logging.info("Worker {} creating metric", rank)
        metric_manager = MetricManage(self.ignore_label)
        metrics = [metric_manager.create_metric(self.metric)]

        logging.info("Worker {} loading data and creating the network symbol", rank)
        train_data_loader = self.train_data_loader
        eval_data_loader = self.eval_data_loader
        network_symbol = self.model.network_symbol(train_data_loader.data_names, train_data_loader.label_names)

        # create bucket model
        logging.info("Worker {} creating bucket model", rank)
        model = mx.mod.BucketingModule(network_symbol, default_bucket_key=train_data_loader.default_bucket_key,
                                       context=devices)
        logging.info("Worker {} begin training", rank)
        # run
        model.fit(train_data_loader,
                  begin_epoch=self.load_epoch if self.load_epoch > 0 else 0,
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
