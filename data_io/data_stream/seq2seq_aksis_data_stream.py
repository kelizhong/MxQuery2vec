# coding=utf-8
from base.base_seq2seq_data_stream import BaseSeq2seqDataStream
import codecs
import re
from nltk.tokenize import word_tokenize
import random

class Seq2seqAksisDataStream(BaseSeq2seqDataStream):
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

    def __init__(self, data_files, encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label=0, floor=-1,
                 dtype='float32'):
        super(Seq2seqAksisDataStream, self).__init__(encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label, dtype)
        self.data_files = data_files
        self.floor = floor

    def data_generator(self):
        for filename in self.data_files:
            with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        line = line.strip().lower()
                        items = re.split(r'\t+', line)
                        if len(items) == 7 and len(items[2]) and len(items[6]) and self.is_hit(items[3]):
                            query = word_tokenize(items[2])
                            title = word_tokenize(items[6])
                            yield query, title
                    except:
                        pass

    def is_hit(self, score):
        return self.floor < -1 or float(score) > random.uniform(self.floor, 1)