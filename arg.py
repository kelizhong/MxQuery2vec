import argparse
import os
from conf.customArgType import IntegerType, LoggerLevelType
def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq model for query2vec')
    subparsers = parser.add_subparsers(help='train or test')

    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")

    train_parser.add_argument('-sny', '--source-num-layers', default=3, type=int,
                        help='number of layers for the source LSTM recurrent neural network')
    train_parser.add_argument('-snh', '--source-num-hidden', default=512, type=int,
                        help='number of hidden units in the source neural network')
    train_parser.add_argument('-ses', '--source-embed-size', default=150, type=int,
                        help='source embedding size ')

    train_parser.add_argument('-tny', '--target-num-layers', default=3, type=int,
                        help='number of layers for the target LSTM recurrent neural network')
    train_parser.add_argument('-tnh', '--target-num-hidden', default=512, type=int,
                        help='number of hidden units in the target neural network')
    train_parser.add_argument('-tes', '--target-embed-size', default=150, type=int,
                        help='target embedding size ')

    train_parser.add_argument('-do', '--dropout', default=0.0, type=float,
                        help='dropout is the probability to ignore the neuron outputs')
    train_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                        default=0, type=int)
    train_parser.add_argument('-mp', '--model-prefix', default=os.path.join(os.getcwd(), 'data', 'model', 'query2vec'), type=str,
                        help='the experiment name, this is also the prefix for the parameters file')
    train_parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                        help='learning rate of the stochastic gradient descent')
    train_parser.add_argument('-lrf', '--lr-factor', default=1, type=float,
                        help='the ratio to reduce lr on each step')
    train_parser.add_argument('-tms', '--train-max-samples', default=2000000, type=int,
                        help='the max sample num for training')
    train_parser.add_argument('-mom', '--momentum', type=float, default=0.9, help='momentum for sgd')
    train_parser.add_argument('-sexb', '--show-every-x-batch', dest='show_every_x_batch', help='show progress for every x batches',
                        default=5, type=int)
    train_parser.add_argument('-ne', '--num-epoch', dest='num_epoch', help='end epoch of query2vec training',
                        default=100000, type=int)
    train_parser.add_argument('-opt', '--optimizer', type=str,
                        help='the optimizer type')
    train_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                        help='batch size for each databatch')


    train_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu'], help='define define mode, (default: %(default)s)',
                        default='cpu')
    train_parser.add_argument('-d', '--devices', type=str,
                        help='the devices will be used, e.g "0,1,2,3"')
    train_parser.add_argument('-ll', '--log-level', choices=['debug', 'info', 'warn', 'error'], default='info', type=LoggerLevelType,
                        help='Log level on console (default: __DEFAULT__).')
    train_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                        help='the frequency to printout the training verbose information')
    train_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    train_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                        help='log network parameters every N iters if larger than 0')



    train_parser.add_argument('--iterations', default=1, type=int,
                        help='number of iterations (epoch)')

    train_parser.add_argument('-pd', '--params-dir', default='params', type=str,
                        help='the directory to store the parameters of the training')
    train_parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')

    train_parser.add_argument('-eti', '--enc-test-input', type=str,
                        help='the file name of the encoder input for testing')
    train_parser.add_argument('-dti', '--dec-test-input', type=str,
                        help='the file name of the decoder input for testing')

    train_parser.add_argument('-tw', '--top-words', default=80, type=int,
                        help='the top percentile of word count to retain in the vocabulary')





    train_parser.add_argument('-wll', '--work-load_-ist', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)

    train_parser.add_argument('-wd','--weight-decay', type=float, default=0.0005, help='weight decay for sgd')
    train_parser.add_argument('--factor-step', type=int, default=50000, help='the step used for lr factor')
    train_parser.add_argument('--monitor', action='store_true', default=False,
                        help='if true, then will use monitor debug')
    train_parser.add_argument('-b', '--bucket', nargs=2, action='append')

    train_parser.add_argument('train_source_path', type=str,
                        help='the file name of the encoder input for training')
    train_parser.add_argument('train_target_path', type=str,
                        help='the file name of the decoder input for training')
    train_parser.add_argument('-vocab', '--vocabulary-path', default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'), type=str,
                        help='vocabulary with he most common words')

    return parser
if __name__ == "__main__":
    print(parse_args().parse_args())