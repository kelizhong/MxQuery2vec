# -*- coding: utf-8 -*-
"""
Sequence 2 sequence for Query2Vec

"""

import os
import sys
from conf.customArgType import DirectoryType
from conf.customArgAction import AppendTupleWithoutDefault
import argparse
from utils.log_util import set_up_logger_handler_with_file
import logging
import signal


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq query2vec for query2vec')
    parser.add_argument('--log-conf-path', default=os.path.join(os.getcwd(), 'configure', 'logger.conf'),
                        type=DirectoryType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('--log-qualname', choices=['root', 'query2vec', 'seq2seq_data_zmq'],
                        default='root',
                        help='Log qualname on console (default: __DEFAULT__).')
    parser.add_argument('--metric-interval', default=6, type=int,
                        help='metric reporting frequency is set by seconds param')
    subparsers = parser.add_subparsers(help='train vocabulary')

    q2v_aksis_ventiliator_parser = subparsers.add_parser("q2v_aksis_ventiliator")
    q2v_aksis_ventiliator_parser.set_defaults(action='q2v_aksis_ventiliator')

    q2v_aksis_ventiliator_parser.add_argument('data_dir', type=str,
                                              help='the file name of the encoder input for training')
    q2v_aksis_ventiliator_parser.add_argument('vocabulary_path',
                                              default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                                              type=str,
                                              help='vocabulary with he most common words')
    q2v_aksis_ventiliator_parser.add_argument('-ap', '--action-patterns', nargs=2, action=AppendTupleWithoutDefault,
                                              default=[('*add', -1), ('*search', 0.5), ('*click', 0.4),
                                                       ('*purchase', -1)])
    q2v_aksis_ventiliator_parser.add_argument('--ip-addr', type=str, help='ip address')
    q2v_aksis_ventiliator_parser.add_argument('--port', type=str, help='zmq port')
    q2v_aksis_ventiliator_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                              help='batch size for each databatch')
    q2v_aksis_ventiliator_parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                                              default=[(3, 10), (3, 20), (5, 20), (7, 30)])
    q2v_aksis_ventiliator_parser.add_argument('--top-words', default=40000, type=int,
                                              help='the max sample num for training')
    return parser.parse_args()


def signal_handler(signal, frame):
    logging.info('Stop!!!')
    sys.exit(0)


def set_up_logger():
    set_up_logger_handler_with_file(args.log_conf_path, args.log_qualname)


if __name__ == "__main__":
    args = parse_args()
    set_up_logger()
    print(args)
    signal.signal(signal.SIGINT, signal_handler)
    if args.action == 'q2v_aksis_ventiliator':
        from data_io.distribute_stream.aksis_data_pipeline import AksisDataPipeline

        p = AksisDataPipeline(args.data_dir, args.vocabulary_path, args.top_words, args.action_patterns,
                              args.batch_size, args.buckets)
        p.start_all()
