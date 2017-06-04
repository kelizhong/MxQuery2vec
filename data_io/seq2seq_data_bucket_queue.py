# coding=utf-8
# pylint: disable=too-few-public-methods
"""seq2seq bucket queue that store train data and generate batch data"""
from collections import deque
import numpy as np


class Seq2seqDataBucketQueue(object):
    """Data stream base class for seq2seq model.

    Parameters
    ----------
        buckets : list of int
            size of data buckets.
        batch_size : int
            batch_size of data
        ignore_label : int
            key for ignore label, the label value will be ignored during backward in
            softmax. Recommend to set 0
        dtype : str, default 'float32'
            data type
    Notes
    -----
    - For query2vec, the vocabulary in encoder is the same with the vocabulary in decoder.
        The vocabulary is from all the corpus including encoder corpus(search query) and
        decoder corpus(asin information)
    """

    def __init__(self, buckets, batch_size, embedding_size=128, ignore_label=0, dtype='float32'):
        self.buckets = buckets
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.ignore_label = ignore_label
        self.dtype = dtype
        self.bucket_queue = dict()
        self._init_queue()

    def _init_queue(self):
        """initialize the queue for each bucket to store tuple(encoder_sentence_id,
        decoder_sentence_id, label_id)"""
        for bucket_key in self.buckets:
            self.bucket_queue.setdefault(bucket_key, deque())

    def put(self, encoder_sentence_id, decoder_sentence_id, label_id):
        """add the data into the bucket queue, return the bucket and queue if the
        queue reach the batch size else return None
        """
        if len(encoder_sentence_id) == 0 or len(decoder_sentence_id) == 0 or len(label_id) == 0:
            return None

        bucket = self._decide_which_bucket(encoder_sentence_id, decoder_sentence_id)
        if bucket:
            self.bucket_queue[bucket].append((encoder_sentence_id, decoder_sentence_id, label_id))
            if len(self.bucket_queue[bucket]) >= self.batch_size:
                return self._get_batch_data_from_queue(bucket, self.bucket_queue[bucket])
        return None

    def _get_batch_data_from_queue(self, bucket, queue):
        """get the batch data from bucket queue and convert the data into numpy format"""
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        ignore_label = self.ignore_label
        dtype = self.dtype
        assert len(queue) >= batch_size, "size of {} queue less than batch size {}".format(bucket, batch_size)
        encoder_data = np.full((batch_size, bucket[0], embedding_size), ignore_label, dtype=dtype)
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

    def _decide_which_bucket(self, encoder, decoder):
        for i, bkt in enumerate(self.buckets):
            if bkt[0] >= len(encoder) and bkt[1] >= len(decoder):
                return bkt
        return None
