# -*- coding: utf-8 -*-
"""
Sequence 2 sequence for Query2Vec

"""

import os
from conf.customArgType import LoggerLevelType, DirectoryType, FileType
from conf.customArgAction import AppendTupleWithoutDefault
import argparse
from common.constant import special_words
from setting import project_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq query2vec for query2vec')

    parser.add_argument('-lp', '--log-path', default=os.path.join(os.getcwd(), 'data', 'logs'),
                        type=DirectoryType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('-ll', '--log-level', choices=['debug', 'info', 'warn', 'error'], default='info',
                        type=LoggerLevelType,
                        help='Log level on console (default: __DEFAULT__).')
    subparsers = parser.add_subparsers(help='train vocabulary')

    q2v_trainer_parser = subparsers.add_parser("train_query2vec", help='train query2vec', add_help=False)
    q2v_trainer_parser.set_defaults(action='train_query2vec')
    q2v_vocab_parser = subparsers.add_parser("query2vec_vocab")
    q2v_vocab_parser.set_defaults(action='query2vec_vocab')
    w2v_trainer_parser = subparsers.add_parser("train_word2vec")
    w2v_trainer_parser.set_defaults(action='train_word2vec')
    w2v_dumper_parser = subparsers.add_parser("dump_word2vec")
    w2v_dumper_parser.set_defaults(action='dump_word2vec')
    q2v_ventiliator_parser = subparsers.add_parser("q2v_ventiliator")
    q2v_ventiliator_parser.set_defaults(action='q2v_ventiliator')

    # model parameter
    q2v_trainer_parser.add_argument('-sln', '--encoder-layer-num', default=1, type=int,
                                    help='number of layers for the encoder LSTM recurrent neural network')
    q2v_trainer_parser.add_argument('-shun', '--encoder-hidden-unit-num', default=5, type=int,
                                    help='number of hidden units in the neural network for encoder')
    q2v_trainer_parser.add_argument('-es', '--embed-size', default=5, type=int,
                                    help='embedding size ')

    q2v_trainer_parser.add_argument('-tln', '--decoder-layer-num', default=1, type=int,
                                    help='number of layers for the decoder LSTM recurrent neural network')
    q2v_trainer_parser.add_argument('-thun', '--decoder-hidden-unit-num', default=5, type=int,
                                    help='number of hidden units in the neural network for decoder')

    q2v_trainer_parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                                    default=[(3, 10), (3, 20), (5, 20), (7, 30)])

    # train parameter

    q2v_trainer_parser.add_argument('-do', '--dropout', default=0.0, type=float,
                                    help='dropout is the probability to ignore the neuron outputs')
    q2v_trainer_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained query2vec',
                                    type=int, default=-1)
    q2v_trainer_parser.add_argument('-mp', '--model-prefix', default='query2vec',
                                    type=str,
                                    help='the experiment name, this is also the prefix for the parameters file')
    q2v_trainer_parser.add_argument('-pd', '--model-path',
                                    default=os.path.join(os.getcwd(), 'data', 'query2vec', 'model'),
                                    type=DirectoryType,
                                    help='the directory to store the parameters of the training')

    q2v_trainer_parser.add_argument('-tms', '--train-max-samples', default=20000, type=int,
                                    help='the max sample num for training')
    q2v_trainer_parser.add_argument('-db', '--disp-batches', dest='disp_batches',
                                    help='show progress for every n batches',
                                    default=1, type=int)
    q2v_trainer_parser.add_argument('-ne', '--num-epoch', dest='num_epoch', help='end epoch of query2vec training',
                                    default=100000, type=int)

    q2v_trainer_parser.add_argument('-bs', '--batch-size', default=3, type=int,
                                    help='batch size for each databatch')

    # optimizer parameter
    q2v_trainer_parser.add_argument('-opt', '--optimizer', type=str, default='Adadelta',
                                    help='the optimizer type')
    q2v_trainer_parser.add_argument('-cg', '--clip-gradient', type=float, default=5.0,
                                    help='clip gradient in range [-clip_gradient, clip_gradient]')
    # mxnet parameter
    q2v_trainer_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu', 'gpu_auto'],
                                    help='define define mode, (default: %(default)s)',
                                    default='cpu')
    q2v_trainer_parser.add_argument('-d', '--devices', type=str, default='0',
                                    help='the devices will be used, e.g "0,1,2,3"')

    q2v_trainer_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                                    help='the frequency to printout the training verbose information')

    q2v_trainer_parser.add_argument('-scf', '--save-checkpoint-freq', default=100, type=int,
                                    help='the frequency to save checkpoint')
    q2v_trainer_parser.add_argument('--save_checkpoint-x-batch',
                                    help='save checkpoint for every x batches',
                                    default=1000, type=int)

    q2v_trainer_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                                    default='local', type=str)
    q2v_trainer_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                                    help='log network parameters every N iters if larger than 0')
    q2v_trainer_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    q2v_trainer_parser.add_argument('--ignore-label', dest='ignore_label', help='ignore label',
                                    default=0, type=int)
    q2v_trainer_parser.add_argument('--hosts-num', dest='hosts_num', help='the number of the hosts',
                                    default=1, type=int)
    q2v_trainer_parser.add_argument('--workers-num', dest='workers_num', help='the number of the workers',
                                    default=1, type=int)
    q2v_trainer_parser.add_argument('--word2vec-path', dest='word2vec_path', help='the number of the workers',
                                    type=str)

    # data parameter
    q2v_trainer_parser.add_argument('--encoder-train-data-path', type=str,
                                    help='the file name of the encoder input for training')
    q2v_trainer_parser.add_argument('--decoder-train-data-path', type=str,
                                    help='the file name of the decoder input for training')
    q2v_trainer_parser.add_argument('--vocabulary-path',
                                    default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                                    type=str,
                                    help='vocabulary with he most common words')
    q2v_trainer_parser.add_argument('--ip-addr', type=str, help='ip address')
    q2v_trainer_parser.add_argument('--port', type=str, help='zmq port')

    # vocabulary parameter
    q2v_vocab_parser.add_argument('-tw', '--top-words', default=40000, type=int,
                                  help='the words with the top frequency to retain in the vocabulary')

    q2v_vocab_parser.add_argument('--overwrite', action='store_true', help='overwrite earlier created files, also forces the\
                        program not to reuse count files')
    q2v_vocab_parser.add_argument('files', nargs='+',
                                  help='the corpus input files')
    q2v_vocab_parser.add_argument('-vf', '--vocab-file',
                                  type=FileType,
                                  default=os.path.join(os.path.dirname(__file__), 'data', 'vocabulary', 'vocab.pkl'),
                                  help='the file with the words which are the most command words in the corpus')

    # word2vec parameter


    # mxnet parameter
    w2v_trainer_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu', 'gpu_auto'],
                                    help='define define mode, (default: %(default)s)',
                                    default='cpu')
    w2v_trainer_parser.add_argument('-d', '--devices', type=str, default='0',
                                    help='the devices will be used, e.g "0,1,2,3"')

    w2v_trainer_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                                    help='the frequency to printout the training verbose information')

    w2v_trainer_parser.add_argument('-scf', '--save-checkpoint-freq', default=1, type=int,
                                    help='the frequency to save checkpoint')

    w2v_trainer_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                                    default='local', type=str)
    w2v_trainer_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                                    help='log network parameters every N iters if larger than 0')
    w2v_trainer_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    w2v_trainer_parser.add_argument('-wll', '--work-load-ist', dest='work_load_list',
                                    help='work load for different devices',
                                    default=None, type=list)
    w2v_trainer_parser.add_argument('--hosts-num', dest='hosts_num', help='the number of the hosts',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('--workers-num', dest='workers_num', help='the number of the workers',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('-db', '--disp-batches', dest='disp_batches',
                                    help='show progress for every n batches',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                                    type=int, default=-1)
    w2v_trainer_parser.add_argument('-r', '--rank', dest='rank', help='epoch of pretrained model',
                                    type=int, default=0)
    w2v_trainer_parser.add_argument('-mp', '--model-prefix', default='word2vec',
                                    type=str,
                                    help='the experiment name, this is also the prefix for the parameters file')
    w2v_trainer_parser.add_argument('-pd', '--model-path',
                                    default=os.path.join(os.getcwd(), 'data', 'word2vec', 'model'),
                                    type=DirectoryType,
                                    help='the directory to store the parameters of the training')

    # optimizer parameter
    w2v_trainer_parser.add_argument('-opt', '--optimizer', type=str, default='AdaGrad',
                                    help='the optimizer type')
    w2v_trainer_parser.add_argument('-cg', '--clip-gradient', type=float, default=5.0,
                                    help='clip gradient in range [-clip_gradient, clip_gradient]')
    w2v_trainer_parser.add_argument('--wd', type=float, default=0.00001,
                                    help='weight decay for sgd')
    w2v_trainer_parser.add_argument('--mom', dest='momentum', type=float, default=0.9,
                                    help='momentum for sgd')
    w2v_trainer_parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                                    help='initial learning rate')

    # model parameter
    w2v_trainer_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                    help='batch size for each databatch')
    w2v_trainer_parser.add_argument('-es', '--embed-size', default=128, type=int,
                                    help='embedding size ')
    w2v_trainer_parser.add_argument('-ws', '--window-size', default=2, type=int,
                                    help='window size ')

    # word2vec data parameter
    w2v_trainer_parser.add_argument('corpus_data_path', type=str,
                                    help='the file name of the corpus')
    w2v_trainer_parser.add_argument('vocabulary_save_path', type=str,
                                    help='the file name of the corpus')

    # word2vec dumper parameter
    w2v_dumper_parser.add_argument('-mp', '--model-prefix', default='word2vec',
                                   type=str,
                                   help='the experiment name, this is also the prefix for the parameters file')
    w2v_dumper_parser.add_argument('-pd', '--model-path',
                                   default=os.path.join(os.getcwd(), 'data', 'word2vec', 'model'),
                                   type=DirectoryType,
                                   help='the directory to store the parameters of the training')
    w2v_dumper_parser.add_argument('-r', '--rank', default=0,
                                   type=int,
                                   help='the experiment name, this is also the prefix for the parameters file')

    w2v_dumper_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                                   type=int, default=1)
    w2v_dumper_parser.add_argument('w2v_vocabulary_path', type=str,
                                   help='the file name of the corpus')
    w2v_dumper_parser.add_argument('w2v_save_path', type=str,
                                   help='the file name of the corpus')

    q2v_ventiliator_parser.add_argument('encoder_train_data_path', type=str,
                                    help='the file name of the encoder input for training')
    q2v_ventiliator_parser.add_argument('decoder_train_data_path', type=str,
                                    help='the file name of the decoder input for training')
    q2v_ventiliator_parser.add_argument('vocabulary_path',
                                    default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                                    type=str,
                                    help='vocabulary with he most common words')
    q2v_ventiliator_parser.add_argument('--ip-addr', type=str, help='ip address')
    q2v_ventiliator_parser.add_argument('--port', type=str, help='zmq port')
    q2v_ventiliator_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                    help='batch size for each databatch')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print project_dir
    print(args)
    if args.action == 'train_query2vec':
        from query2vec.query2vec_trainer import Query2vecTrainer, mxnet_parameter, optimizer_parameter, model_parameter, \
            data_parameter

        mxnet_para = mxnet_parameter(kv_store=args.kv_store, hosts_num=args.hosts_num, workers_num=args.workers_num,
                                     device_mode=args.device_mode, devices=args.devices,
                                     disp_batches=args.disp_batches, monitor_interval=args.monitor_interval,
                                     log_level=args.log_level, log_path=args.log_path,
                                     save_checkpoint_freq=args.save_checkpoint_freq,
                                     model_path_prefix=os.path.join(args.model_path, args.model_prefix),
                                     enable_evaluation=args.enable_evaluation, ignore_label=args.ignore_label,
                                     load_epoch=args.load_epoch, train_max_samples=args.train_max_samples,
                                     save_checkpoint_x_batch=args.save_checkpoint_x_batch)

        optimizer_para = optimizer_parameter(optimizer=args.optimizer, clip_gradient=args.clip_gradient)

        model_para = model_parameter(encoder_layer_num=args.encoder_layer_num,
                                     encoder_hidden_unit_num=args.encoder_hidden_unit_num,
                                     encoder_embed_size=args.embed_size, encoder_dropout=args.dropout,
                                     decoder_layer_num=args.decoder_layer_num,
                                     decoder_hidden_unit_num=args.decoder_hidden_unit_num,
                                     decoder_embed_size=args.embed_size, decoder_dropout=args.dropout,
                                     batch_size=args.batch_size, buckets=args.buckets)

        data_para = data_parameter(encoder_train_data_path=args.encoder_train_data_path,
                                   decoder_train_data_path=args.decoder_train_data_path,
                                   vocabulary_path=args.vocabulary_path, ip_addr=args.ip_addr, port=args.port,
                                   word2vec_path=args.word2vec_path)

        trainer = Query2vecTrainer(
            mxnet_para=mxnet_para, optimizer_para=optimizer_para, model_para=model_para, data_para=data_para)
        trainer.train()
    elif args.action == 'query2vec_vocab':
        from vocabulary.vocabulary import Vocab

        vocab = Vocab(args.files, args.vocab_file, top_words=args.top_words,
                      special_words=special_words,
                      log_path=args.log_path, log_level=args.log_level, overwrite=args.overwrite)
        vocab.create_dictionary()
    elif args.action == 'train_word2vec':
        from word2vec.word2vec_trainer import Word2vecTrainer, mxnet_parameter, optimizer_parameter, model_parameter

        mxnet_para = mxnet_parameter(kv_store=args.kv_store, hosts_num=args.hosts_num, workers_num=args.workers_num,
                                     device_mode=args.device_mode, devices=args.devices,
                                     disp_batches=args.disp_batches, monitor_interval=args.monitor_interval,
                                     log_level=args.log_level, log_path=args.log_path,
                                     save_checkpoint_freq=args.save_checkpoint_freq,
                                     model_path_prefix=os.path.join(args.model_path, args.model_prefix),
                                     enable_evaluation=args.enable_evaluation,
                                     load_epoch=args.load_epoch)

        optimizer_para = optimizer_parameter(optimizer=args.optimizer, learning_rate=args.learning_rate, wd=args.wd,
                                             momentum=args.momentum)

        model_para = model_parameter(embed_size=args.embed_size, batch_size=args.batch_size,
                                     window_size=args.window_size)
        Word2vecTrainer(data_path=args.corpus_data_path, vocabulary_save_path=args.vocabulary_save_path,
                        mxnet_para=mxnet_para,
                        optimizer_para=optimizer_para,
                        model_para=model_para).train()
    elif args.action == 'dump_word2vec':
        from word2vec.word2vec_dumper import W2vDumper

        W2vDumper(w2v_model_path=os.path.join(args.model_path, args.model_prefix), vocab_path=args.w2v_vocabulary_path,
                  w2v_save_path=args.w2v_save_path, rank=args.rank, load_epoch=args.load_epoch).dumper()
    elif args.action == 'q2v_ventiliator':
        from data_io.distribute_stream.seq2seq_data_ventilator import Seq2seqDataVentilator
        from data_io.data_stream.seq2seq_data_stream import Seq2seqDataStream
        from utils.data_util import load_pickle_object

        vocab = load_pickle_object(args.vocabulary_path)
        s = Seq2seqDataStream(args.encoder_train_data_path,
                              args.decoder_train_data_path, vocab,
                              vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], args.batch_size)
        a = Seq2seqDataVentilator(s)
        a.produce()
