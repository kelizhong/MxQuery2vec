import mxnet as mx
import bisect
import random
import numpy as np
from common.constant import bos_word, eos_word


class BucketSentenceIter(mx.io.DataIter):
    """Simple bucketing iterator for language model.
    Label for each step is constructed from data of
    next step.
    Parameters
    ----------
    sentences : list of list of int
        encoded sentences
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
    def __init__(self, encoder_path, decoder_path, encoder_vocab, decoder_vocab, buckets, sentence2id=None, read_data=None, invalid_label=-1,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 layout='NTC'):
        super(BucketSentenceIter, self).__init__()

        encoder_sentences, decoder_sentences = read_data(encoder_path, decoder_path)

        assert len(encoder_sentences) == len(decoder_sentences)
        num_of_data = len(encoder_sentences)
        for i in range(num_of_data):
            encoder = encoder_sentences[i]
            decoder = [bos_word] + decoder_sentences[i]
            label = decoder_sentences[i] + [eos_word]
            encoder_sentence = sentence2id(encoder, encoder_vocab)
            decoder_sentence = sentence2id(decoder, decoder_vocab)
            label_id = sentence2id(label, decoder_vocab)
            if len(encoder_sentence) == 0 or len(decoder_sentence) == 0:
                continue
            for j, bkt in enumerate(buckets):
                if bkt[0] >= len(encoder) and bkt[1] >= len(decoder):
                    self.encoder_data[j].append(encoder_sentence)
                    self.decoder_data[j].append(decoder_sentence)
                    self.label_data[j].append(label_id)
                    break
                    # we just ignore the sentence it is longer than the maximum
                    # bucket size here
        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(sentences):
            buck = bisect.bisect_left(buckets, len(sent))
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sent)] = sent
            self.data[buck].append(buff)

        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)

        if self.major_axis == 0:
            self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
            self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
        elif self.major_axis == 1:
            self.provide_data = [(data_name, (self.default_bucket_key, batch_size))]
            self.provide_label = [(label_name, (self.default_bucket_key, batch_size))]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

        self.nddata = []
        self.ndlabel = []
        for buck in self.data:
            label = np.empty_like(buck)
            label[:, :-1] = buck[:, 1:]
            label[:, -1] = self.invalid_label
            self.nddata.append(mx.ndarray.array(buck, dtype=self.dtype))
            self.ndlabel.append(mx.ndarray.array(label, dtype=self.dtype))

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            data = self.nddata[i][j:j+self.batch_size].T
            label = self.ndlabel[i][j:j+self.batch_size].T
        else:
            data = self.nddata[i][j:j+self.batch_size]
            label = self.ndlabel[i][j:j+self.batch_size]

        return mx.DataBatch([data], [label],
                         bucket_key=self.buckets[i],
                         provide_data=[(self.data_name, data.shape)],
                         provide_label=[(self.label_name, label.shape)])