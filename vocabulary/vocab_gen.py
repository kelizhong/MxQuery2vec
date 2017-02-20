# coding=utf-8
"""Count the frequency of unique words in a file.
"""
import re
import logging
import pickle
import os
from collections import Counter
import time


class vocab(object):
    def __init__(self, corpus_files, vocab_file, most_commond_words_file, special_words=dict(), most_commond_words_num=40000, loglevel=logging.INFO, logdir='./', overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_file = vocab_file
        self.most_commond_words_file = most_commond_words_file
        self.special_words = special_words
        self.most_commond_words_num = most_commond_words_num
        self.overwrite = overwrite
        self.logdir = logdir
        self._init_log(loglevel)

    def _init_log(self, loglevel):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=loglevel, datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(os.path.join(self.logdir, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
        logging.root.addHandler(file_handler)

    def _words_gen(self, file):
        """Return each word in a line."""
        for line in file:
            words = re.split('\s+', line.strip())
            for word in words:
                yield word

    def _safe_pickle(self, obj, filename):
        if os.path.isfile(filename) and not self.overwrite:
            logging.warning("Not saving %s, already exists." % (filename))
        else:
            if os.path.isfile(filename):
                logging.info("Overwriting %s." % filename)
            else:
                logging.info("Saving to %s." % filename)
            with open(filename, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def create_dictionary(self):
        """Start execution of word-frequency."""
        global_counter = Counter()
        for file in self.corpus_files:
            logging.info("Counting words in %s" % file)
            with open(file) as f:
                counter = Counter(self._words_gen(f))
                logging.info("%d unique words in %s with a total of %d words."
                        % (len(counter), file, sum(counter.values())))
            global_counter.update(counter)
            logging.info("Finish counting. %d unique words, a total of %d words in all files."
                    % (len(global_counter), sum(counter.values())))

        words_num = len(global_counter)
        special_words_num = len(self.special_words)

        assert words_num > len(self.special_words), "the size of total words must be larger than the size of special_words"

        logging.info("store vocalbulary file")
        self._safe_pickle(global_counter, self.vocab_file)

        assert self.most_commond_words_num > len(
            self.special_words), "the value of most_commond_words_num must be larger than the size of special_words"

        vocab_count = global_counter.most_common(self.most_commond_words_num - special_words_num)
        print(len(vocab_count))
        vocab = {}
        idx = special_words_num + 1
        for word, _ in vocab_count:
            if word not in self.special_words:
                vocab[word] = idx
                idx += 1
        vocab.update(self.special_words)
        logging.info("store vocalbulary with most_commond_words file")
        self._safe_pickle(vocab, self.most_commond_words_file)

