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
    def __init__(self, encoder_path, decoder_path, encoder_vocab, decoder_vocab, buckets, batch_size, max_sentence_num=sys.maxsize):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.buckets = buckets
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.max_sentence_num = max_sentence_num
        self.data_gen = self.data_generator()
        self.batch_size = batch_size
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
            #print(encoder_words, decoder_words)
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
        assert len(queue) >= batch_size, "size of {} queue less than batch size {}".format(bucket, batch_size)
        encoder_data = np.zeros((batch_size, bucket[0]))
        encoder_mask_data = np.zeros((batch_size, bucket[0]))
        decoder_data = np.zeros((batch_size, bucket[1]))
        decoder_mask_data = np.zeros((batch_size, bucket[1]))
        label_data = np.zeros((batch_size, bucket[1]))
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
                          vocab, 'fds')
    for encoder_data, encoder_mask_data, decoder_data, decoder_mask_data, label_data, bucket in s:
        print(encoder_data, decoder_data)
