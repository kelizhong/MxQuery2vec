# -*- coding: utf-8 -*-
"""
Sequence 2 sequence for Query2Vec

"""

import os
from conf.customArgType import LoggerLevelType, DirectoryType, FileType
from conf.customArgAction import AppendTupleWithoutDefault
import argparse
from common.constant import special_words


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq model for query2vec')

    parser.add_argument('-lp', '--log-path', default=os.path.join(os.getcwd(), 'data', 'logs'),
                        type=DirectoryType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('-ll', '--log-level', choices=['debug', 'info', 'warn', 'error'], default='info',
                        type=LoggerLevelType,
                        help='Log level on console (default: __DEFAULT__).')
    subparsers = parser.add_subparsers(help='train vocabulary')

    train_parser = subparsers.add_parser("train", help='train model', add_help=False)
    train_parser.set_defaults(action='train')
    vocab_parser = subparsers.add_parser("vocab")
    vocab_parser.set_defaults(action='vocab')
    w2v_parser = subparsers.add_parser("word2vec")
    w2v_parser.set_defaults(action='word2vec')

    # model parameter
    train_parser.add_argument('-sln', '--encoder-layer-num', default=1, type=int,
                              help='number of layers for the encoder LSTM recurrent neural network')
    train_parser.add_argument('-shun', '--encoder-hidden-unit-num', default=3, type=int,
                              help='number of hidden units in the neural network for encoder')
    train_parser.add_argument('-es', '--embed-size', default=128, type=int,
                              help='embedding size ')

    train_parser.add_argument('-tln', '--decoder-layer-num', default=1, type=int,
                              help='number of layers for the decoder LSTM recurrent neural network')
    train_parser.add_argument('-thun', '--decoder-hidden-unit-num', default=3, type=int,
                              help='number of hidden units in the neural network for decoder')

    train_parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                              default=[(3, 10), (3, 20), (5, 20), (7, 30)])

    # train parameter

    train_parser.add_argument('-do', '--dropout', default=0.0, type=float,
                              help='dropout is the probability to ignore the neuron outputs')
    train_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                              type=int)
    train_parser.add_argument('-mp', '--model-prefix', default='query2vec',
                              type=str,
                              help='the experiment name, this is also the prefix for the parameters file')
    train_parser.add_argument('-pd', '--model-path', default=os.path.join(os.getcwd(), 'data', 'model'),
                              type=DirectoryType,
                              help='the directory to store the parameters of the training')

    train_parser.add_argument('-tms', '--train-max-samples', default=20000, type=int,
                              help='the max sample num for training')
    train_parser.add_argument('-db', '--disp-batches', dest='disp_batches',
                              help='show progress for every n batches',
                              default=1, type=int)
    train_parser.add_argument('-ne', '--num-epoch', dest='num_epoch', help='end epoch of query2vec training',
                              default=100000, type=int)

    train_parser.add_argument('-bs', '--batch-size', default=2, type=int,
                              help='batch size for each databatch')

    # optimizer parameter
    train_parser.add_argument('-opt', '--optimizer', type=str, default='Adadelta',
                              help='the optimizer type')
    train_parser.add_argument('-cg', '--clip-gradient', type=float, default=5.0,
                              help='clip gradient in range [-clip_gradient, clip_gradient]')
    # mxnet parameter
    train_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu', 'gpu_auto'],
                              help='define define mode, (default: %(default)s)',
                              default='cpu')
    train_parser.add_argument('-d', '--devices', type=str, default='0',
                              help='the devices will be used, e.g "0,1,2,3"')

    train_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                              help='the frequency to printout the training verbose information')

    train_parser.add_argument('-scf', '--save-checkpoint-freq', default=100, type=int,
                              help='the frequency to save checkpoint')

    train_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                              default='local', type=str)
    train_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                              help='log network parameters every N iters if larger than 0')
    train_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    train_parser.add_argument('-eti', '--enc-test-input', type=str,
                              help='the file name of the encoder input for testing')
    train_parser.add_argument('-dti', '--dec-test-input', type=str,
                              help='the file name of the decoder input for testing')

    train_parser.add_argument('-wll', '--work-load-ist', dest='work_load_list', help='work load for different devices',
                              default=None, type=list)
    train_parser.add_argument('--ignore-label', dest='ignore_label', help='ignore label',
                              default=0, type=int)
    train_parser.add_argument('--hosts-num', dest='hosts_num', help='the number of the hosts',
                              default=1, type=int)
    train_parser.add_argument('--workers-num', dest='workers_num', help='the number of the workers',
                              default=1, type=int)

    # data parameter
    train_parser.add_argument('encoder_train_data_path', type=str,
                              help='the file name of the encoder input for training')
    train_parser.add_argument('decoder_train_data_path', type=str,
                              help='the file name of the decoder input for training')
    train_parser.add_argument('vocabulary_path', default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                              type=str,
                              help='vocabulary with he most common words')

    # vocabulary parameter
    vocab_parser.add_argument('-tw', '--top-words', default=40000, type=int,
                              help='the words with the top frequency to retain in the vocabulary')

    vocab_parser.add_argument('--overwrite', action='store_true', help='overwrite earlier created files, also forces the\
                        program not to reuse count files')
    vocab_parser.add_argument('files', nargs='+',
                              help='the corpus input files')
    vocab_parser.add_argument('-vf', '--vocab-file',
                              type=FileType,
                              default=os.path.join(os.path.dirname(__file__), 'data', 'vocabulary', 'vocab.pkl'),
                              help='the file with the words which are the most command words in the corpus')

    # word2vec parameter


    # mxnet parameter
    w2v_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu', 'gpu_auto'],
                            help='define define mode, (default: %(default)s)',
                            default='cpu')
    w2v_parser.add_argument('-d', '--devices', type=str, default='0',
                            help='the devices will be used, e.g "0,1,2,3"')

    w2v_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                            help='the frequency to printout the training verbose information')

    w2v_parser.add_argument('-scf', '--save-checkpoint-freq', default=1, type=int,
                            help='the frequency to save checkpoint')

    w2v_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                            default='local', type=str)
    w2v_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                            help='log network parameters every N iters if larger than 0')
    w2v_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    w2v_parser.add_argument('-wll', '--work-load-ist', dest='work_load_list', help='work load for different devices',
                            default=None, type=list)
    w2v_parser.add_argument('--ignore-label', dest='ignore_label', help='ignore label',
                            default=0, type=int)
    w2v_parser.add_argument('--hosts-num', dest='hosts_num', help='the number of the hosts',
                            default=1, type=int)
    w2v_parser.add_argument('--workers-num', dest='workers_num', help='the number of the workers',
                            default=1, type=int)
    w2v_parser.add_argument('-db', '--disp-batches', dest='disp_batches',
                            help='show progress for every n batches',
                            default=1, type=int)
    w2v_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                            type=int)
    w2v_parser.add_argument('-mp', '--model-prefix', default='query2vec',
                            type=str,
                            help='the experiment name, this is also the prefix for the parameters file')
    w2v_parser.add_argument('-pd', '--model-path', default=os.path.join(os.getcwd(), 'data', 'model'),
                            type=DirectoryType,
                            help='the directory to store the parameters of the training')
    w2v_parser.add_argument('-tms', '--train-max-samples', default=20000, type=int,
                            help='the max sample num for training')

    # optimizer parameter
    w2v_parser.add_argument('-opt', '--optimizer', type=str, default='AdaGrad',
                            help='the optimizer type')
    w2v_parser.add_argument('-cg', '--clip-gradient', type=float, default=5.0,
                            help='clip gradient in range [-clip_gradient, clip_gradient]')
    w2v_parser.add_argument('--wd', type=float, default=0.00001,
                            help='weight decay for sgd')
    w2v_parser.add_argument('--mom', dest='momentum', type=float, default=0.9,
                            help='momentum for sgd')
    w2v_parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                            help='initial learning rate')

    # model parameter
    w2v_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                            help='batch size for each databatch')
    w2v_parser.add_argument('-es', '--embed-size', default=128, type=int,
                            help='embedding size ')
    w2v_parser.add_argument('-ws', '--window-size', default=2, type=int,
                            help='window size ')

    # data parameter
    w2v_parser.add_argument('data_path', type=str,
                            help='the file name of the corpus')
    w2v_parser.add_argument('vocabulary_save_path', type=str,
                            help='the file name of the corpus')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(args)
    if args.action == 'train':
        from model.query2vec_trainer import Query2vecTrainer, mxnet_parameter, optimizer_parameter, model_parameter

        mxnet_para = mxnet_parameter(kv_store=args.kv_store, hosts_num=args.hosts_num, workers_num=args.workers_num,
                                     device_mode=args.device_mode, devices=args.devices,
                                     disp_batches=args.disp_batches, monitor_interval=args.monitor_interval,
                                     log_level=args.log_level, log_path=args.log_path,
                                     save_checkpoint_freq=args.save_checkpoint_freq,
                                     model_path_prefix=os.path.join(args.model_path, args.model_prefix),
                                     enable_evaluation=args.enable_evaluation, ignore_label=args.ignore_label,
                                     load_epoch=args.load_epoch, train_max_samples=args.train_max_samples)

        optimizer_para = optimizer_parameter(optimizer=args.optimizer, clip_gradient=args.clip_gradient)

        model_para = model_parameter(encoder_layer_num=args.encoder_layer_num,
                                     encoder_hidden_unit_num=args.encoder_hidden_unit_num,
                                     encoder_embed_size=args.embed_size, encoder_dropout=args.dropout,
                                     decoder_layer_num=args.decoder_layer_num,
                                     decoder_hidden_unit_num=args.decoder_hidden_unit_num,
                                     decoder_embed_size=args.embed_size, decoder_dropout=args.dropout,
                                     batch_size=args.batch_size, buckets=args.buckets)

        trainer = Query2vecTrainer(encoder_train_data_path=args.encoder_train_data_path,
                                   decoder_train_data_path=args.decoder_train_data_path,
                                   vocabulary_path=args.vocabulary_path,
                                   mxnet_para=mxnet_para, optimizer_para=optimizer_para, model_para=model_para)
        trainer.train()
    elif args.action == 'vocab':
        from vocabulary.vocabulary import Vocab

        vocab = Vocab(args.files, args.vocab_file, top_words=args.top_words,
                      special_words=special_words,
                      log_path=args.log_path, log_level=args.log_level, overwrite=args.overwrite)
        vocab.create_dictionary()
    elif args.action == 'word2vec':
        from word2vec.word2vec_trainer import Word2vecTrainer, mxnet_parameter, optimizer_parameter, model_parameter

        mxnet_para = mxnet_parameter(kv_store=args.kv_store, hosts_num=args.hosts_num, workers_num=args.workers_num,
                                     device_mode=args.device_mode, devices=args.devices,
                                     disp_batches=args.disp_batches, monitor_interval=args.monitor_interval,
                                     log_level=args.log_level, log_path=args.log_path,
                                     save_checkpoint_freq=args.save_checkpoint_freq,
                                     model_path_prefix=os.path.join(args.model_path, args.model_prefix),
                                     enable_evaluation=args.enable_evaluation,
                                     load_epoch=args.load_epoch, train_max_samples=args.train_max_samples)

        optimizer_para = optimizer_parameter(optimizer=args.optimizer, learning_rate=args.learning_rate, wd=args.wd,
                                             momentum=args.momentum)

        model_para = model_parameter(embed_size=args.embed_size, batch_size=args.batch_size,
                                     window_size=args.window_size)
        Word2vecTrainer(data_path=args.data_path, vocabulary_save_path=args.vocabulary_save_path, mxnet_para=mxnet_para, optimizer_para=optimizer_para,
                        model_para=model_para).train()
