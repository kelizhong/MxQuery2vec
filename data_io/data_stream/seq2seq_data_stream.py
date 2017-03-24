# coding=utf-8
import sys
import codecs
from nltk.tokenize import word_tokenize
import itertools
from collections import deque
from common.constant import bos_word, eos_word
from utils.data_util import sentence2id, load_pickle_object
import numpy as np


class Seq2seqDataStream(object):
    """masked bucketing iterator for seq2seq model. This class is only used for test

    Parameters
    ----------
    encoder_path : str
        encoder corpus path
    decoder_path: str
        decoder corpus path
    encoder_vocab: dict
        vocabulary from encoder corpus.
    decoder_vocab: dict
        vocabulary from decoder corpus.
    buckets : list of int
        size of data buckets.
    batch_size : int
        batch_size of data
    ignore_label : int
        key for ignore label, the label value will be ignored during backward in softmax. Recommend to set 0
    dtype : str, default 'float32'
        data type
    max_sentence_num: int
        the max size of sentence to read
    Notes
    -----
    - For query2vec, the vocabulary in encoder is the same with the vocabulary in decoder.
        The vocabulary is from all the corpus including encoder corpus(search query) adn decoder corpus(asin information)
    """

    def __init__(self, encoder_path, decoder_path, encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label=0,
                 dtype='float32', max_sentence_num=sys.maxsize):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.buckets = buckets
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.max_sentence_num = max_sentence_num
        self.data_gen = self.data_generator()
        self.batch_size = batch_size
        self.ignore_label = ignore_label
        self.dtype = dtype
        self.bucket_queue = dict()
        self._init_queue()

    def _init_queue(self):
        for bucket_key in self.buckets:
            self.bucket_queue.setdefault(bucket_key, deque())
        self.curr_queue = self.bucket_queue.get(self.buckets[0])

    def data_generator(self):
        with codecs.open(self.encoder_path) as encoder, codecs.open(self.decoder_path) as decoder:
            for encoder_line, decoder_line in itertools.izip(itertools.islice(encoder, self.max_sentence_num),
                                                             itertools.islice(decoder, self.max_sentence_num)):
                try:
                    encoder_line = encoder_line.strip().lower()
                    decoder_line = decoder_line.strip().lower()
                    encoder_line = encoder_line.decode('utf-8')
                    decoder_line = decoder_line.decode('utf-8')
                    if len(encoder_line) and len(decoder_line):
                        encoder_words = word_tokenize(encoder_line)
                        decoder_words = word_tokenize(decoder_line)
                    yield encoder_words, decoder_words
                except Exception, e:
                    pass

    def __iter__(self):
        return self

    def next(self):

        while True:
            encoder_words, decoder_words = self.data_gen.next()
            # print(encoder_words, decoder_words)
            bucket, queue = self.add_to_queue(encoder_words, decoder_words)
            if queue:
                return self.get_batch_data_from_queue(bucket, queue)

    def add_to_queue(self, encoder_words, decoder_words):
        encoder_sentence_id, decoder_sentence_id, label_id = self.convert_data_to_id(encoder_words, decoder_words)
        if len(encoder_sentence_id) == 0 or len(decoder_sentence_id) == 0:
            pass

        bucket = self.decide_which_bucket(encoder_sentence_id, decoder_sentence_id)

        if bucket:
            self.bucket_queue[bucket].append((encoder_sentence_id, decoder_sentence_id, label_id))
            if len(self.bucket_queue[bucket]) >= self.batch_size:
                return bucket, self.bucket_queue[bucket]
        return None, None

    def convert_data_to_id(self, encoder_words, decoder_words):
        encoder = encoder_words
        decoder = [bos_word] + decoder_words
        label = decoder_words + [eos_word]
        encoder_sentence_id = sentence2id(encoder, self.encoder_vocab)
        decoder_sentence_id = sentence2id(decoder, self.decoder_vocab)
        label_id = sentence2id(label, self.decoder_vocab)
        return encoder_sentence_id, decoder_sentence_id, label_id

    def get_batch_data_from_queue(self, bucket, queue):
        batch_size = self.batch_size
        ignore_label = self.ignore_label
        dtype = self.dtype
        assert len(queue) >= batch_size, "size of {} queue less than batch size {}".format(bucket, batch_size)
        encoder_data = np.full((batch_size, bucket[0]), ignore_label, dtype=dtype)
        encoder_mask_data = np.full((batch_size, bucket[0]), ignore_label, dtype=dtype)
        decoder_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        decoder_mask_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        label_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        for i in xrange(batch_size):
            encoder_sentence_id, decoder_sentence_id, label_id = queue.popleft()
            encoder_data[i, :len(encoder_sentence_id)] = encoder_sentence_id
            encoder_mask_data[i, :len(encoder_sentence_id)] = 1
            decoder_data[i, :len(decoder_sentence_id)] = decoder_sentence_id
            decoder_mask_data[i, :len(decoder_sentence_id)] = 1
            label_data[i, :len(label_id)] = label_id
        return encoder_data, encoder_mask_data, decoder_data, decoder_mask_data, label_data, bucket

    def decide_which_bucket(self, encoder, decoder):
        for i, bkt in enumerate(self.buckets):
            if bkt[0] >= len(encoder) and bkt[1] >= len(decoder):
                return bkt
        return None

    def reset(self):
        self.data_gen = self.data_generator()


if __name__ == '__main__':
    vocab = load_pickle_object('../data/vocabulary/vocab.pkl')
    s = Seq2seqDataStream('../data/query2vec/train_corpus/small.enc', '../data/query2vec/train_corpus/small.dec', vocab,
                          vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], 3)
    for encoder_data, encoder_mask_data, decoder_data, decoder_mask_data, label_data, bucket in s:
        print(encoder_data, decoder_data)
