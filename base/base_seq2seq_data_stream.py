# coding=utf-8
import abc
from collections import deque
from common.constant import bos_word, eos_word
from utils.data_util import sentence2id
import numpy as np


class BaseSeq2seqDataStream(object):
    """Data stream base class for seq2seq model.

    Parameters
    ----------
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
    Notes
    -----
    - For query2vec, the vocabulary in encoder is the same with the vocabulary in decoder.
        The vocabulary is from all the corpus including encoder corpus(search query) adn decoder corpus(asin information)
    """

    def __init__(self, encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label=0,
                 dtype='float32'):
        self.buckets = buckets
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.data_gen = self.data_generator()
        self.batch_size = batch_size
        self.ignore_label = ignore_label
        self.dtype = dtype
        self.bucket_queue = dict()
        self._init_queue()

    def _init_queue(self):
        """initialize the queue for each bucket to store tuple(encoder_sentence_id, decoder_sentence_id, label_id)"""
        for bucket_key in self.buckets:
            self.bucket_queue.setdefault(bucket_key, deque())

    @abc.abstractmethod
    def data_generator(self):
        """generate the data for seq2seq model, including two parts(encoder_words, decoder_words)
           encoder_words, decoder_words is a list object which has been segmented .e.g.
           encoder line(maybe query/post): mens running shoes
           decoder line(maybe title/response): Nike Men's Shox NZ SE Black/White/Paramount Blue Running Shoe 10.5 Men U
           then
           encoder_words: [mens, running, shoes]
           decoder_words: [Nike, Men's, Shox, NZ, SE, Black, /, White, /, Paramount, Blue, Running, Shoe, 10.5, Men, U]
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):

        while True:
            # get data from generator
            encoder_words, decoder_words = self.data_gen.next()
            # add the data include corresponding queue, return the bucket
            # and queue if size of the queue reach the batch size
            bucket, queue = self.add_to_bucket_queue(encoder_words, decoder_words)
            # if the returned queue is not None, return the batch size data to zmq ventiliator(for distributed version)
            # or just return data to mxnet iter interface(this is usually for test)
            if queue:
                return self.get_batch_data_from_queue(bucket, queue)

    def add_to_bucket_queue(self, encoder_words, decoder_words):
        """add the data into the bucket queue, return the bucket and queue if the queue reach the batch size
            else return None,None
        """
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
        """convert the data into id which represent the index for word in vocabulary"""
        encoder = encoder_words
        decoder = [bos_word] + decoder_words
        label = decoder_words + [eos_word]
        encoder_sentence_id = sentence2id(encoder, self.encoder_vocab)
        decoder_sentence_id = sentence2id(decoder, self.decoder_vocab)
        label_id = sentence2id(label, self.decoder_vocab)
        return encoder_sentence_id, decoder_sentence_id, label_id

    def get_batch_data_from_queue(self, bucket, queue):
        """get the batch data from bucket queue and convert the data into numpy format"""
        batch_size = self.batch_size
        ignore_label = self.ignore_label
        dtype = self.dtype
        assert len(queue) >= batch_size, "size of {} queue less than batch size {}".format(bucket, batch_size)
        encoder_data = np.full((batch_size, bucket[0]), ignore_label, dtype=dtype)
        encoder_mask_data = np.full((batch_size, bucket[0]), ignore_label, dtype=dtype)
        decoder_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        decoder_mask_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        label_data = np.full((batch_size, bucket[1]), ignore_label, dtype=dtype)
        # encoder_data = np.zeros((batch_size, bucket[0]))
        # encoder_mask_data = np.zeros((batch_size, bucket[0]))
        # decoder_data = np.zeros((batch_size, bucket[1]))
        # decoder_mask_data = np.zeros((batch_size, bucket[1]))
        # label_data = np.zeros((batch_size, bucket[1]))
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
        """reset the data generator for reading the data cyclicly"""
        self.data_gen = self.data_generator()
