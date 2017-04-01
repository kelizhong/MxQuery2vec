# coding=utf-8
import codecs
import itertools
import sys
from utils.data_util import tokenize
from base.base_seq2seq_data_stream import BaseSeq2seqDataStream
from utils.data_util import load_pickle_object
import logging


class Seq2seqDataStream(BaseSeq2seqDataStream):
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
        super(Seq2seqDataStream, self).__init__(encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label, dtype)
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.max_sentence_num = max_sentence_num

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
                        encoder_words = tokenize(encoder_line)
                        decoder_words = tokenize(decoder_line)
                    if encoder_words and decoder_words:
                        yield encoder_words, decoder_words
                except Exception as e:
                    logging.error(e)


if __name__ == '__main__':
    vocab = load_pickle_object('../data/vocabulary/vocab.pkl')
    s = Seq2seqDataStream('../data/query2vec/train_corpus/small.enc', '../data/query2vec/train_corpus/small.dec', vocab,
                          vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], 3)
    for encoder_data, encoder_mask_data, decoder_data, decoder_mask_data, label_data, bucket in s:
        print(encoder_data, decoder_data)
