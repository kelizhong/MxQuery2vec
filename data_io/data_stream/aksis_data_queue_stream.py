# coding=utf-8
from base.base_seq2seq_data_stream import BaseSeq2seqDataStream
import Queue
import random
import logging
from utils.data_util import extract_query_title_score_from_aksis_data


class Seq2seqAksisDataStream(BaseSeq2seqDataStream):
    """Data stream for aksis data for seq2seq model. This class is for distributed training

    Parameters
    ----------
    data_files : list
        data files for training, the format of every line in these files is:
        MarketplaceId\tAsin\tKeyword\tScore\t ActionType\tDate\tasin_title
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
    floor: float
        this value is used to generate random value between [floor, 1]
    Notes
    -----
    - For query2vec, the vocabulary in encoder is the same with the vocabulary in decoder.
        The vocabulary is from all the corpus including encoder corpus(search query) adn decoder corpus(asin information)
    """

    def __init__(self, queue, encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label=0, sample_floor=-1,
                 dtype='float32'):
        super(Seq2seqAksisDataStream, self).__init__(encoder_vocab, decoder_vocab, buckets, batch_size, ignore_label, dtype)
        self.queue = queue
        self.sample_floor = float(sample_floor)

    def data_generator(self):
        while True:
            try:
                data = self.queue.get(timeout=1)
                yield data
            except Queue.Empty:
                raise StopIteration

    def is_hit(self, score):
        """sample function to decide whether the data should be trained, not sample if floor less than 0"""
        return self.sample_floor < 0 or float(score) > random.uniform(self.sample_floor, 1)