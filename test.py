import mxnet as mx
import numpy as np
import argparse
from collections import OrderedDict
from utils.data_utils import load_vocab, sentence2id, word2id
from inference.inference_model import BiS2SInferenceModel_mask
import logging
import os
import time
from conf.customArgType import LoggerLevelType, DirectoryType
from conf.customArgAction import AppendTupleWithoutDefault
from utils.data_utils import get_stop_words
from nltk.tokenize import wordpunct_tokenize
from common import constant

def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq model for query2vec')

    parser.add_argument('-lp', '--log-path', default=os.path.join(os.getcwd(), 'data', 'logs'),
                        type=DirectoryType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('-ll', '--log-level', choices=['debug', 'info', 'warn', 'error'], default='info',
                        type=LoggerLevelType,
                        help='Log level on console (default: __DEFAULT__).')
    parser.add_argument('vocabulary_path', default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                        type=str,
                        help='vocabulary with he most common words')
    parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                        type=int)
    parser.add_argument('-mp', '--model-prefix', default='query2vec',
                        type=str,
                        help='the experiment name, this is also the prefix for the parameters file')
    parser.add_argument('-pd', '--model-path', default=os.path.join(os.getcwd(), 'data', 'model'),
                        type=DirectoryType,
                        help='the directory to store the parameters of the training')

    # model parameter
    parser.add_argument('-sln', '--source-layer-num', default=1, type=int,
                        help='number of layers for the source LSTM recurrent neural network')
    parser.add_argument('-shun', '--source-hidden-unit-num', default=64, type=int,
                        help='number of hidden units in the neural network for encoder')
    parser.add_argument('-es', '--embed-size', default=32, type=int,
                        help='embedding size ')

    parser.add_argument('-tln', '--target-layer-num', default=1, type=int,
                        help='number of layers for the target LSTM recurrent neural network')
    parser.add_argument('-thun', '--target-hidden-unit-num', default=64, type=int,
                        help='number of hidden units in the neural network for decoder')

    parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                        default=[(5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (60, 60)])
    parser.add_argument('-swd', '--stop-words-dir',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'stop_words'),
                        help='stop words file directory')
    return parser.parse_args()


def get_inference_models(buckets, arg_params, source_vocab_size, target_vocab_size, ctx, batch_size):
    # build an inference model
    model_buckets = OrderedDict()
    for bucket in buckets:
        model_buckets[bucket] = BiS2SInferenceModel_mask(s_num_lstm_layer=args.source_layer_num, s_seq_len=bucket[0],
                                                         s_vocab_size=source_vocab_size,
                                                         s_num_hidden=args.source_hidden_unit_num,
                                                         s_num_embed=args.embed_size,
                                                         s_dropout=0,
                                                         t_num_lstm_layer=args.target_layer_num,
                                                         t_vocab_size=target_vocab_size,
                                                         t_num_hidden=args.target_hidden_unit_num,
                                                         t_num_embed=args.embed_size,
                                                         t_num_label=target_vocab_size, t_dropout=0,
                                                         arg_params=arg_params,
                                                         use_masking=True,
                                                         ctx=ctx, batch_size=batch_size)
    return model_buckets


def get_bucket_model(model_buckets, input_len):
    for bucket, m in model_buckets.items():
        if bucket[0] >= input_len:
            return bucket[0], m
    return None, None


# make input from char
def MakeInput(sentence, vocab, unroll_len, data_arr, mask_arr):
    idx = sentence2id(sentence, vocab, stop_words)
    tmp = np.zeros((1, unroll_len))
    mask = np.zeros((1, unroll_len))
    for i in range(min(len(idx), unroll_len)):
        tmp[0][i] = idx[i]
        mask[0][i] = 1
    data_arr[:] = tmp
    mask_arr[:] = mask


def MakeTargetInput(char, vocab, arr):
    idx = word2id(char, vocab)
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp


def MakeOutput(prob, vocab):
    idx = np.argmax(prob, axis=1)[0]
    try:
        char = vocab[idx]
    except:
        char = ''
    return char


def reponse(max_decode_len, sentence, model_buckets, source_vocab, target_vocab, revert_vocab,
                  target_ndarray):
    input_length = len(sentence)
    unroll_len, cur_model = get_bucket_model(model_buckets, input_length)
    input_ndarray = mx.nd.zeros((1, unroll_len))
    mask_ndarray = mx.nd.zeros((1, unroll_len))
    output = [constant.bos_word]
    MakeInput(sentence, source_vocab, unroll_len, input_ndarray, mask_ndarray)
    last_encoded, _ = cur_model.encode(input_ndarray,
                                       mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput(output[-1], target_vocab, target_ndarray)
        prob = cur_model.decode_forward(last_encoded, target_ndarray,
                                        i == 0)
        next_char = MakeOutput(prob, revert_vocab)
        if next_char == constant.eos_word:
            break
        output.append(next_char)
    return output[1:]


# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic


def test_use_model_param(str):
    # load vocabulary
    vocab = load_vocab(args.vocabulary_path)
    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(os.path.join(args.model_path, args.model_prefix), args.load_epoch)
    vocab_size = len(vocab)
    logging.info('vocab size: {0}'.format(vocab_size))
    target_ndarray = mx.nd.zeros((1,), ctx=mx.cpu())
    revert_vocab = MakeRevertVocab(vocab)

    buckets = args.buckets
    model_buckets = get_inference_models(buckets, arg_params, len(vocab), len(vocab),
                                         mx.cpu(), batch_size=1)
    en = reponse(15, str, model_buckets, vocab, vocab,
                       revert_vocab,
                       target_ndarray)
    del model_buckets
    en = ' '.join(en)
    return en


def load_model_buckets(vocab_size):
    buckets = args.buckets
    model_buckets = get_inference_models(buckets, arg_params, vocab_size, vocab_size,
                                         mx.cpu(), batch_size=1)
    return model_buckets


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=args.log_level,
                        datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(args.log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    args.load_epoch = 28
    stop_words = get_stop_words(args.stop_words_dir, 'english')

    # load vocabulary
    vocab = load_vocab(args.vocabulary_path)
    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(os.path.join(args.model_path, args.model_prefix), args.load_epoch)
    vocab_size = len(vocab)
    logging.info('vocab size: {0}'.format(vocab_size))
    revert_vocab = MakeRevertVocab(vocab)
    buckets = args.buckets
    model_buckets = load_model_buckets(vocab_size)
    with open('./data/train_corpus/train.enc') as f:
        for line in f:
            line = line.strip()
            print(line)
            target_ndarray = mx.nd.zeros((1,), ctx=mx.cpu())
            en = reponse(15, wordpunct_tokenize(line), model_buckets, vocab, vocab,
                               revert_vocab,
                               target_ndarray)
            en = ' '.join(en)
            print(en)
    a = wordpunct_tokenize("At least, the cat comes back.")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("You're asking me out.  That's so cute. What's your name again?")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("Right.")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("Hello.")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("good bye")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("I vote we go back to The Slaughtered Lamb.")
    print(a)
    print(test_use_model_param(a))
