# -*- coding: utf-8 -*-
"""
Sequence 2 sequence for Query2Vec

"""

import logging
import yaml
import yamlordereddictloader
import os
import clg
import time
from conf.customYamlType import IntegerType, LoggerLevelType
# sys.path.append('.')
# sys.path.append('..')


def parseArgs(config_path):
    clg.TYPES.update({'Integer': IntegerType, 'LoggerLevel': LoggerLevelType})
    cmd = clg.CommandLine(yaml.load(open(config_path),
                                    Loader=yamlordereddictloader.Loader))
    args = cmd.parse()

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=args.loglevel, datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(args.logdir, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)

    for arg, value in args:
        print("  %s: %s" % (arg, value))
    return args


if __name__ == "__main__":
    args = parseArgs('./conf/config.yml')
    if args.command0 == 'train':
        logging.info('In train mode.')
        from model.trainer import trainer

        trainer(train_source_path=args.train_source_path, train_target_path=args.train_target_path, vocabulary_path=args.vocabulary_path) \
            .set_model_parameter(s_layer_num=args.s_layer_num, s_hidden_unit=args.s_hidden_unit,
                                 s_embed_size=args.s_embed_size, t_layer_num=args.t_layer_num,
                                 t_hidden_unit=args.t_hidden_unit, t_embed_size=args.t_embed_size, buckets=args.buckets)\
            .set_train_parameter(s_dropout=args.s_dropout, t_dropout=args.t_dropout, load_epoch=args.load_epoch,
                                 model_prefix=args.model_prefix, device_model=args.device_model, devices=args.devices,
                                 lr_factor=args.lr_factor,\
                                 lr=args.lr, train_max_samples=args.train_max_samples, momentum=args.momentum,
                                 show_every_x_batch=args.show_every_x_batch, num_epoch=args.num_epoch,
                                 optimizer=args.optimizer, batch_size=args.batch_size)\
            .set_mxnet_parameter(loglevel=args.loglevel)\
            .train()
    elif args.command0 =='vocab':
        from vocabulary.vocab_gen import vocab
        vocab(args.files, args.vocab_file_path, args.most_commond_words_file_path, special_words=args.special_words, logdir=args.logdir)\
            .create_dictionary()
