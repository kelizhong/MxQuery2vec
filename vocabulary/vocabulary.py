# coding=utf-8
import logging
import os
from collections import Counter
import time
from utils.pickle_util import save_obj_pickle
from utils.data_util import words_gen


class Vocab(object):
    """
    Create vocabulary file (if it does not exist yet) from data file.
    Data file should have one sentence per line.
    Each sentence will be tokenized.
    Vocabulary contains the most-frequent tokens up to top_words.
    We write it to vocab_file in pickle format.
    special_words will be added into the vocabulary file
    Parameters
    ----------
    corpus_files: list
        corpus files list that will be used to create vocabulary
    vocab_file: str
        vocab file name where the vocabulary will be created
    special_words: dict
        special words, e.g.<s>, </s>, <unk>
    top_words: int
        limit on the size of the created vocabulary
    log_level: level name, e.g.INFO, ERROR
        log level
    log_path: str
        log path where the log will be saved
    overwrite: bool
        whether to overwrite the existed vocabulary
    """
    def __init__(self, corpus_files, vocab_file, special_words=dict(), top_words=40000,
                 log_level=logging.INFO,
                 log_path='./', overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_file = vocab_file
        self.top_words = top_words
        self.special_words = special_words
        self.overwrite = overwrite
        self.log_path = log_path
        self._init_log(log_level)

    def _init_log(self, log_level):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=log_level,
                            datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(os.path.join(self.log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
        logging.root.addHandler(file_handler)

    def create_dictionary(self):
        """Start execution of word-frequency."""
        global_counter = Counter()
        for filename in self.corpus_files:
            logging.info("Counting words in %s" % filename)
            counter = Counter(words_gen(filename))
            logging.info("%d unique words in %s with a total of %d words."
                         % (len(counter), filename, sum(counter.values())))
            global_counter.update(counter)

        logging.info("Finish counting. %d unique words, a total of %d words in all files."
                     % (len(global_counter), sum(counter.values())))

        words_num = len(global_counter)
        special_words_num = len(self.special_words)

        assert words_num > len(
            self.special_words), "the size of total words must be larger than the size of special_words"

        assert self.top_words > len(
            self.special_words), "the value of most_commond_words_num must be larger than the size of special_words"

        vocab_count = global_counter.most_common(self.top_words - special_words_num)
        vocab = {}
        idx = special_words_num + 1
        for word, _ in vocab_count:
            if word not in self.special_words:
                vocab[word] = idx
                idx += 1
        vocab.update(self.special_words)
        logging.info("store vocabulary with most_common_words file, vocabulary size: " + str(len(vocab)))
        save_obj_pickle(vocab, self.vocab_file, self.overwrite)
