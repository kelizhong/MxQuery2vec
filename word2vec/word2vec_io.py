import random
import math
import mxnet as mx
from utils.data_util import words_gen
from collections import Counter
from common.constant import unk_word, bos_word, eos_word
from utils.pickle_util import save_obj_pickle


def build_word2vec_dataset(filename, vocabulary_save_path, vocabulary_size):
    freq = [[unk_word, -1], [bos_word, -1], [eos_word, -1]]
    counter = Counter(words_gen(filename))
    assert vocabulary_size > len(freq), "vocabulary_size must be larger than {}".format(len(freq))
    freq.extend(counter.most_common(vocabulary_size - len(freq)))
    vocabulary = dict()
    for word, _ in freq:
        vocabulary[word] = len(vocabulary)
    data = list()
    unk_count = 0
    bos_count = 0
    eos_count = 0
    for word in words_gen(filename, bos=bos_word, eos=eos_word):
        bos_count = bos_count + 1 if word == bos_word is not None else bos_count
        eos_count = eos_count + 1 if word == eos_word is not None else eos_count
        if word in vocabulary:
            index = vocabulary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    freq[0][1] = unk_count
    freq[1][1] = bos_count
    freq[2][1] = eos_count

    negative = []
    for i, v in enumerate(freq):
        count = v[1]
        if i == 0:
            continue
        count = int(math.pow(count * 1.0, 0.75))
        negative += [i for _ in range(count)]

    reverse_vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))

    # save vocabulary
    save_obj_pickle(vocabulary, vocabulary_save_path)

    return data, negative, vocabulary, freq, reverse_vocabulary


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class Word2vecDataIter(mx.io.DataIter):
    def __init__(self, filename, vocabulary_save_path, batch_size, num_label, data_name='data', label_name='label',
                 label_weight_name='label_weight'):
        super(Word2vecDataIter, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, _, _ = build_word2vec_dataset(filename, vocabulary_save_path, 1000000)
        self.vocab_size = 1 + len(self.vocab)
        self.num_label = num_label
        self.data_name = data_name
        self.label_name = label_name
        self.label_weight_name = label_weight_name
        self.provide_data = [(data_name, (batch_size, num_label - 1))]
        self.provide_label = [(label_name, (self.batch_size, num_label)),
                              (label_weight_name, (self.batch_size, num_label))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    @property
    def data_names(self):
        return [self.data_name]

    @property
    def label_names(self):
        return [self.label_name] + [self.label_weight_name]

    def __iter__(self):
        batch_data = []
        batch_label = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.data) - self.num_label - start, self.num_label):
            context = self.data[i: i + self.num_label / 2] \
                      + self.data[i + 1 + self.num_label / 2: i + self.num_label]
            target_word = self.data[i + self.num_label / 2]
            if target_word in [0, 1, 2]:
                continue
            target = [target_word] \
                     + [self.sample_ne() for _ in range(self.num_label - 1)]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            batch_data.append(context)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(self.data_names, data_all, self.label_names, label_all)

    def reset(self):
        pass
