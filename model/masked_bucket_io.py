import sys
import numpy as np
import mxnet as mx
from common.constant import bos_word, eos_word


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class MaskedBucketSentenceIter(mx.io.DataIter):
    def __init__(self, encoder_path, decoder_path, encoder_vocab, decoder_vocab,
                 buckets, batch_size,
                 encoder_init_states, decoder_init_states,
                 encoder_data_name='encoder_data', encoder_mask_name='encoder_mask',
                 decoder_data_name='decoder_data', decoder_mask_name='decoder_mask',
                 label_name='decoder_softmax_label',
                 sentence2id=None, read_data=None, max_read_sample=sys.maxsize):
        super(MaskedBucketSentenceIter, self).__init__()

        self.sentence2id = sentence2id
        encoder_sentences = read_data(encoder_path, max_read_sample)

        decoder_sentences = read_data(decoder_path, max_read_sample)

        assert len(encoder_sentences) == len(decoder_sentences)

        self.encoder_vocab_size = len(encoder_vocab)
        self.decoder_vocab_size = len(decoder_vocab)
        self.encoder_data_name = encoder_data_name
        self.decoder_data_name = decoder_data_name
        self.label_name = label_name

        self.encoder_mask_name = encoder_mask_name
        self.decoder_mask_name = decoder_mask_name

        buckets.sort()
        self.buckets = buckets
        self.encoder_data = [[] for _ in buckets]
        self.decoder_data = [[] for _ in buckets]
        self.label_data = [[] for _ in buckets]
        self.encoder_mask_data = [[] for _ in buckets]
        self.decoder_mask_data = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        num_of_data = len(encoder_sentences)
        for i in range(num_of_data):
            encoder = encoder_sentences[i]
            decoder = [bos_word] + decoder_sentences[i]
            label = decoder_sentences[i] + [eos_word]
            encoder_sentence = self.sentence2id(encoder, encoder_vocab)
            decoder_sentence = self.sentence2id(decoder, decoder_vocab)
            label_id = self.sentence2id(label, decoder_vocab)
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

        # convert data into ndarrays for better speed during training
        encoder_data = [np.zeros((len(x), buckets[i][0])) for i, x in enumerate(self.encoder_data)]
        encoder_mask_data = [np.zeros((len(x), buckets[i][0])) for i, x in enumerate(self.encoder_data)]
        decoder_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.decoder_data)]
        decoder_mask_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.decoder_data)]
        label_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.label_data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.encoder_data[i_bucket])):
                encoder = self.encoder_data[i_bucket][j]
                decoder = self.decoder_data[i_bucket][j]
                label = self.label_data[i_bucket][j]
                encoder_data[i_bucket][j, :len(encoder)] = encoder
                encoder_mask_data[i_bucket][j, :len(encoder)] = 1
                decoder_data[i_bucket][j, :len(decoder)] = decoder
                decoder_mask_data[i_bucket][j, :len(decoder)] = 1
                label_data[i_bucket][j, :len(label)] = label
        self.encoder_data = encoder_data
        self.encoder_mask_data = encoder_mask_data
        self.decoder_data = decoder_data
        self.decoder_mask_data = decoder_mask_data
        self.label_data = label_data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.encoder_data]

        print("Summary of dataset ==================")
        print('Total: {0} in {1} buckets'.format(num_of_data, len(buckets)))
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of {0} : {1} samples".format(bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.encoder_init_states = encoder_init_states
        self.decoder_init_states = decoder_init_states
        self.encoder_init_state_arrays = [mx.nd.zeros(x[1]) for x in encoder_init_states]
        self.decoder_init_state_arrays = [mx.nd.zeros(x[1]) for x in decoder_init_states]

        self.encoder_init_state_names = [x[0] for x in encoder_init_states]
        self.decoder_init_state_names = [x[0] for x in decoder_init_states]

        self.provide_data = [(encoder_data_name, (batch_size, self.default_bucket_key[0])),
                             (encoder_mask_name, (batch_size, self.default_bucket_key[0])),
                             (decoder_data_name, (batch_size, self.default_bucket_key[1])),
                             (decoder_mask_name, (batch_size, self.default_bucket_key[1]))] \
                            + encoder_init_states + decoder_init_states
        self.provide_label = [(label_name, (self.batch_size, self.default_bucket_key[1]))]

    def get_data_names(self):
        return [self.encoder_data_name] + [self.encoder_mask_name] + [self.decoder_data_name] + [self.decoder_mask_name] \
               + self.encoder_init_state_names + self.decoder_init_state_names

    def get_label_names(self):
        return [self.label_name]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.encoder_data)):
            bucket_n_batches.append(len(self.encoder_data[i]) / self.batch_size)
            self.encoder_data[i] = self.encoder_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.encoder_mask_data[i] = self.encoder_mask_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.decoder_data[i] = self.decoder_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.decoder_mask_data[i] = self.decoder_mask_data[i][:int(bucket_n_batches[i] * self.batch_size)]

        ################################################################################
        # >> > a = np.zeros(3, int)
        # >> > a
        # array([0, 0, 0])
        # >> > b = np.zeros(4, int) + 2
        # >> > b
        # array([2, 2, 2, 2])
        # >> > np.hstack([a, b])
        # array([0, 0, 0, 2, 2, 2, 2])
        # >> > c = np.hstack([a, b])
        # >> > np.random.shuffle(c)
        # >> > c
        # array([2, 2, 0, 0, 2, 2, 0])
        ################################################################################
        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.encoder_data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for _ in self.encoder_data]

        # buffer for each batch data
        self.encoder_data_buffer = []
        self.encoder_mask_data_buffer = []
        self.decoder_data_buffer = []
        self.decoder_mask_data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.encoder_data)):
            encoder_data = np.zeros((self.batch_size, self.buckets[i_bucket][0]))
            encoder_mask_data = np.zeros((self.batch_size, self.buckets[i_bucket][0]))
            decoder_data = np.zeros((self.batch_size, self.buckets[i_bucket][1]))
            decoder_mask_data = np.zeros((self.batch_size, self.buckets[i_bucket][1]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket][1]))

            self.encoder_data_buffer.append(encoder_data)
            self.encoder_mask_data_buffer.append(encoder_mask_data)
            self.decoder_data_buffer.append(decoder_data)
            self.decoder_mask_data_buffer.append(decoder_mask_data)
            self.label_buffer.append(label)

    def __iter__(self):
        encoder_init_state_names = [x[0] for x in self.encoder_init_states]
        decoder_init_state_names = [x[0] for x in self.decoder_init_states]

        for i_bucket in self.bucket_plan:
            encoder_data = self.encoder_data_buffer[i_bucket]
            encoder_mask_data = self.encoder_mask_data_buffer[i_bucket]
            decoder_data = self.decoder_data_buffer[i_bucket]
            decoder_mask_data = self.decoder_mask_data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            encoder_data[:] = self.encoder_data[i_bucket][idx]
            encoder_mask_data[:] = self.encoder_mask_data[i_bucket][idx]
            decoder_data[:] = self.decoder_data[i_bucket][idx]
            decoder_mask_data[:] = self.decoder_mask_data[i_bucket][idx]
            label[:] = self.label_data[i_bucket][idx]

            data_all = [mx.nd.array(encoder_data), mx.nd.array(encoder_mask_data)] + \
                       [mx.nd.array(decoder_data), mx.nd.array(decoder_mask_data)] + \
                       self.encoder_init_state_arrays + self.decoder_init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = [self.encoder_data_name, self.encoder_mask_name] + [
                self.decoder_data_name, self.decoder_mask_name] + encoder_init_state_names + decoder_init_state_names
            label_names = [self.label_name]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for _ in self.encoder_data]
