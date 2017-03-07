import mxnet as mx
import numpy as np
import argparse
import random
import bisect
from collections import OrderedDict, namedtuple
from utils.data_utils import load_vocab, sentence2id, word2id
from inference.inference_model import BiS2SInferenceModel_mask
import logging
import yaml
import yamlordereddictloader
import os
import clg
import time
from conf.customArgType import IntegerType, LoggerLevelType, DirectoryType, FileType
from conf.customArgAction import AppendTupleWithoutDefault
from utils.data_utils import get_stop_words
from nltk.tokenize import wordpunct_tokenize
random_sample = False

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
    parser.add_argument('-shun', '--source-hidden-unit-num', default=512, type=int,
                              help='number of hidden units in the neural network for encoder')
    parser.add_argument('-es', '--embed-size', default=128, type=int,
                              help='embedding size ')

    parser.add_argument('-tln', '--target-layer-num', default=1, type=int,
                              help='number of layers for the target LSTM recurrent neural network')
    parser.add_argument('-thun', '--target-hidden-unit-num', default=512, type=int,
                              help='number of hidden units in the neural network for decoder')

    parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                              default=[(3, 10), (3, 20), (5, 20), (7, 30)])
    parser.add_argument('-swd', '--stop-words-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'stop_words'), help='stop words file directory')
    return parser.parse_args()

def get_inference_models(buckets, arg_params, source_vocab_size, target_vocab_size, ctx, batch_size):
    # build an inference model
    model_buckets = OrderedDict()
    for bucket in buckets:
        model_buckets[bucket] = BiS2SInferenceModel_mask(s_num_lstm_layer=args.source_layer_num, s_seq_len=bucket[0],
                                                         s_vocab_size=source_vocab_size,
                                                         s_num_hidden=args.source_hidden_unit_num, s_num_embed=args.embed_size,
                                                         s_dropout=0,
                                                         t_num_lstm_layer=args.target_layer_num, t_seq_len=bucket[1],
                                                         t_vocab_size=target_vocab_size,
                                                         t_num_hidden=args.target_hidden_unit_num, t_num_embed=args.embed_size,
                                                         t_num_label=target_vocab_size, t_dropout=0,
                                                         arg_params=arg_params,
                                                         use_masking=True,
                                                         ctx=ctx, batch_size=batch_size)
    return model_buckets


def get_bucket_model(model_buckets, input_len):
    for bucket, m in model_buckets.items():
        if bucket[0] >= input_len:
            return m
    return None

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

# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample == False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    return char
# helper function for random sample
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

def translate_one(max_decode_len, sentence, model_buckets, unroll_len, source_vocab, target_vocab, revert_vocab,
                  target_ndarray):
    input_length = len(sentence)
    cur_model = get_bucket_model(model_buckets, input_length)
    input_ndarray = mx.nd.zeros((1, unroll_len))
    mask_ndarray = mx.nd.zeros((1, unroll_len))
    output = ['<s>']
    MakeInput(sentence, source_vocab, unroll_len, input_ndarray, mask_ndarray)
    last_encoded, all_encoded = cur_model.encode(input_ndarray,
                                                 mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput(output[-1], target_vocab, target_ndarray)
        prob = cur_model.decode_forward(last_encoded, target_ndarray,
                                                           i == 0)
        next_char = MakeOutput(prob, revert_vocab, random_sample)
        if next_char == '</s>':
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
    buckets = [max(buckets)]
    model_buckets = get_inference_models(buckets, arg_params, len(vocab), len(vocab),
                                         mx.cpu(), batch_size=1)
    en = translate_one(15, str, model_buckets, max(buckets)[0], vocab, vocab,
                       revert_vocab,
                       target_ndarray)
    del model_buckets
    en = ' '.join(en)
    return en


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=args.log_level, datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(args.log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    args.load_epoch = 140
    stop_words = get_stop_words(args.stop_words_dir, 'english')
    a = wordpunct_tokenize("I figured you'd get to the good stuff eventually.")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("what is your name")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("hi")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("hello")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("good bye")
    print(a)
    print(test_use_model_param(a))
    a = wordpunct_tokenize("good morning")
    print(a)
    print(test_use_model_param(a))