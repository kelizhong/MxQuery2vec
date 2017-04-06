# coding=utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""masked bucketing iterator for seq2seq model"""
import codecs
import itertools
import sys
import logging
from utils.data_util import tokenize, convert_data_to_id
from data_io.seq2seq_data_bucket_queue import Seq2seqDataBucketQueue


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
            key for ignore label, the label value will be
            ignored during backward in softmax. Recommend to set 0
        dtype : str, default 'float32'
            data type
        max_sentence_num: int
        the max size of sentence to read
    Notes
    -----
    - For query2vec, the vocabulary in encoder is the same with the
    vocabulary in decoder. The vocabulary is from all the corpus
    including encoder corpus(search query) adn decoder corpus(asin information)
    """

    def __init__(self, encoder_path, decoder_path, encoder_vocab, decoder_vocab,
                 buckets, batch_size, ignore_label=0,
                 dtype='float32', max_sentence_num=sys.maxsize):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.max_sentence_num = max_sentence_num
        self.buckets = buckets
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.data_gen = self.data_generator()
        self.batch_size = batch_size
        self.ignore_label = ignore_label
        self.dtype = dtype
        self.bucket_queue = dict()
        self.queue = self._init_queue()

    def _init_queue(self):
        """initialize the queue for each bucket to store
        tuple(encoder_sentence_id, decoder_sentence_id, label_id)"""
        queue = Seq2seqDataBucketQueue(self.buckets, self.batch_size)
        return queue

    def data_generator(self):
        """generate the data for seq2seq model, including two parts(encoder_words, decoder_words)
           encoder_words, decoder_words is a list object which has been segmented .e.g.
           encoder line(maybe query/post): mens running shoes
           decoder line(maybe title/response): Nike Men's Shox NZ SE Black/White/Paramount Blue
                                               Running Shoe 10.5 Men U
           then
           encoder_words: [mens, running, shoes]
           decoder_words: [Nike, Men's, Shox, NZ, SE, Black, /, White, /, Paramount, Blue,
                          Running, Shoe, 10.5, Men, U]
        """
        with codecs.open(self.encoder_path) as encoder, codecs.open(self.decoder_path) as decoder:
            # pylint: disable=line-too-long
            for encoder_line, decoder_line in itertools.izip(itertools.islice(encoder, self.max_sentence_num),
                                                             itertools.islice(decoder, self.max_sentence_num)):
                try:
                    encoder_line = encoder_line.strip().lower()
                    decoder_line = decoder_line.strip().lower()
                    encoder_line = encoder_line.decode('utf-8')
                    decoder_line = decoder_line.decode('utf-8')
                    if len(encoder_line) and len(decoder_line):
                        encoder_words = tokenize(encoder_line)
                        decoder_words = tokenize(decoder_line)
                    if encoder_words and decoder_words:
                        yield encoder_words, decoder_words
                # pylint: disable=invalid-name
                except Exception as e:
                    logging.error(e)

    def __iter__(self):
        return self

    def next(self):
        """return batch train data(encoder_data, encoder_mask_data,
        decoder_data, decoder_mask_data, label_data, bucket)"""
        while True:
            # get data from generator
            encoder_words, decoder_words = self.data_gen.next()
            # pylint: disable=line-too-long
            encoder_sentence_id, decoder_sentence_id, label_id = convert_data_to_id(encoder_words, decoder_words,
                                                                                    self.encoder_vocab,
                                                                                    self.decoder_vocab)

            data = self.queue.put(encoder_sentence_id, decoder_sentence_id, label_id)
            if data:
                return data

    def reset(self):
        """reset the data generator for reading the data cyclicly"""
        self.data_gen = self.data_generator()
