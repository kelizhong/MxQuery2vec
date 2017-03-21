import mxnet as mx
import numpy as np
import argparse
from collections import OrderedDict
from utils.data_util import load_pickle_object, sentence2id, word2id
from inference.inference_model import  BiSeq2seqInferenceModel
import logging
import os
import time
from conf.customArgType import LoggerLevelType, DirectoryType
from conf.customArgAction import AppendTupleWithoutDefault
from nltk.tokenize import word_tokenize
from common import constant
from numpy import linalg as la
import pickle

def euclidSimilar(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))


def pearsonSimilar(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]


def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


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
    parser.add_argument('-pd', '--model-path', default=os.path.join(os.getcwd(), 'data', 'query2vec/model'),
                        type=DirectoryType,
                        help='the directory to store the parameters of the training')

    # model parameter
    parser.add_argument('-sln', '--source-layer-num', default=1, type=int,
                        help='number of layers for the source LSTM recurrent neural network')
    parser.add_argument('-shun', '--source-hidden-unit-num', default=5, type=int,
                        help='number of hidden units in the neural network for encoder')
    parser.add_argument('-es', '--embed-size', default=5, type=int,
                        help='embedding size ')

    parser.add_argument('-tln', '--target-layer-num', default=1, type=int,
                        help='number of layers for the target LSTM recurrent neural network')
    parser.add_argument('-thun', '--target-hidden-unit-num', default=5, type=int,
                        help='number of hidden units in the neural network for decoder')

    parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWithoutDefault, type=int,
                        default=[(3,10), (3,20), (5,20), (7,30)])
    parser.add_argument('-swd', '--stop-words-dir',
                        default=os.path.join(os.path.dirname(__file__), 'data', 'stop_words'),
                        help='stop words file directory')
    return parser.parse_args()


def get_inference_models(buckets, arg_params, encoder_vocab_size, decoder_vocab_size, ctx, batch_size):
    # build an inference model
    model_buckets = OrderedDict()
    for bucket in buckets:
        model_buckets[bucket] = BiSeq2seqInferenceModel(encoder_layer_num=args.source_layer_num, encoder_seq_len=bucket[0],
                                                      encoder_vocab_size=encoder_vocab_size,
                                                      encoder_hidden_unit_num=args.source_hidden_unit_num,
                                                      encoder_embed_size=args.embed_size,
                                                      encoder_dropout=0,
                                                      decoder_layer_num=args.target_layer_num,
                                                      decoder_vocab_size=decoder_vocab_size,
                                                      decoder_hidden_unit_num=args.target_hidden_unit_num,
                                                      decoder_embed_size=args.embed_size,
                                                      decoder_dropout=0,
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
    idx = sentence2id(sentence, vocab)
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
    last_encoded, all_encoded= cur_model.encode(input_ndarray,
                                       mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput(output[-1], target_vocab, target_ndarray)
        prob = cur_model.decode_forward(last_encoded, all_encoded, target_ndarray,
                                        i == 0)
        next_char = MakeOutput(prob, revert_vocab)
        if next_char == constant.eos_word:
            break
        output.append(next_char)
    return output[1:]


def encode(sentence, model_buckets, source_vocab):
    input_length = len(sentence)
    unroll_len, cur_model = get_bucket_model(model_buckets, input_length)
    input_ndarray = mx.nd.zeros((1, unroll_len))
    mask_ndarray = mx.nd.zeros((1, unroll_len))
    MakeInput(sentence, source_vocab, unroll_len, input_ndarray, mask_ndarray)
    last_encoded, _ = cur_model.encode(input_ndarray,
                                       mask_ndarray)  # last_encoded means the last time step hidden

    return last_encoded

# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic


def test_use_model_param(str):
    # load vocabulary
    vocab = load_pickle_object(args.vocabulary_path)
    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(os.path.join(args.model_path, args.model_prefix), args.load_epoch)
    vocab_size = len(vocab) + 1
    logging.info('vocab size: {0}'.format(vocab_size))
    target_ndarray = mx.nd.zeros((1,), ctx=mx.cpu())
    revert_vocab = MakeRevertVocab(vocab)

    buckets = args.buckets
    model_buckets = get_inference_models(buckets, arg_params, vocab_size, vocab_size,
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

def generate_embeddings(model_buckets, vocab):
    with open('./data/train_corpus/conversation.post') as f, open('./data/meta.txt', 'w+') as meta, open('./data/embedding.pkl', 'wb') as embed:
        i = 0
        embedding_dict = dict()
        for line in f:
            try:
                line = line.strip()
                words = word_tokenize(line)
                embedding = encode(words, model_buckets, vocab).asnumpy()
                embedding_dict.setdefault(i, embedding)
                meta.write(str(i) + '\t' + line + "\n")
                i = i + 1
                if i > 100:
                    break
                if i%100 == 0:
                    print("Count: " + str(i/100))
            except:
                pass
        pickle.dump(embedding_dict, embed, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=args.log_level,
                        datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(args.log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    args.load_epoch = 800

    # load vocabulary
    vocab = load_pickle_object(args.vocabulary_path)
    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(os.path.join(args.model_path, args.model_prefix), args.load_epoch)
    vocab_size = len(vocab) + 1
    logging.info('vocab size: {0}'.format(vocab_size))
    revert_vocab = MakeRevertVocab(vocab)
    buckets = args.buckets
    model_buckets = load_model_buckets(vocab_size)
    #generate_embeddings(model_buckets, vocab)
    #with open('./data/train_corpus/conversation.post') as f:
    #    for line in f:
    #        line = line.strip()
    #        print(line)
    #        target_ndarray = mx.nd.zeros((1,), ctx=mx.cpu())
    #        en = reponse(15, wordpunct_tokenize(line), model_buckets, vocab, vocab,
    #                           revert_vocab,
    #                           target_ndarray)
    #        en = ' '.join(en)
    #        print(en)
    #a = word_tokenize("colonel durnford... william vereker. i hear you 've been seeking officers?")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("hello")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("how are you?")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("what's up?")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("what is the meaning of life?")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("should i kill someone?")
    #print(a)
    #print(test_use_model_param(a))
    #a = word_tokenize("i'm to take the sikali with the main column to the river")
    #print(a)
    #print(test_use_model_param(a))
    #a = encode(word_tokenize("bad"), model_buckets, vocab).asnumpy()
    #b = encode(word_tokenize("thanks"), model_buckets, vocab).asnumpy()
    #print(cosSimilar(a, b))
    #c = encode(word_tokenize("thank you"), model_buckets, vocab).asnumpy()
    #print(cosSimilar(b, c))
    a = encode(word_tokenize("women nike shoe"), model_buckets, vocab).asnumpy()
    b = encode(word_tokenize("iphone"), model_buckets, vocab).asnumpy()
    print(cosSimilar(a, b))
    c = encode(word_tokenize("iphone6"), model_buckets, vocab).asnumpy()
    print(cosSimilar(b, c))

    target_ndarray = mx.nd.zeros((1,), ctx=mx.cpu())
    en = reponse(15, word_tokenize("what is your name"), model_buckets, vocab, vocab,
                       revert_vocab,
                       target_ndarray)
    en = ' '.join(en)
    print(en)
