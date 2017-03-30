# coding=utf-8
import logging
import os
from collections import Counter
import time
from utils.pickle_util import save_obj_pickle
from vocabulary.ventilator import VentilatorProcess
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess


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

    def __init__(self, corpus_files, vocab_save_path, process_num=10, ip='127.0.0.1', ventilator_port='5555', collector_port='5556',
                 log_level=logging.INFO,
                 log_path='./', overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_save_path = vocab_save_path
        self.process_num = process_num
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
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
        process_pool = []
        v = VentilatorProcess(self.corpus_files, self.ip, self.ventilator_port)
        v.start()
        process_pool.append(v)
        for i in xrange(self.process_num):
            w = WorkerProcess(self.ip, self.ventilator_port, self.collector_port, name='WorkerProcess_{}'.format(i))
            w.start()
            process_pool.append(w)
        c = CollectorProcess(self.ip, self.collector_port)
        counter = Counter(c.collect())
        self._terminate_process(process_pool)
        logging.info("Finish counting. %d unique words, a total of %d words in all files."
                     % (len(counter), sum(counter.values())))

        #words_num = len(counter)
        #special_words_num = len(self.special_words)

        #assert words_num > len(
        #    self.special_words), "the size of total words must be larger than the size of special_words"

        #assert self.top_words > len(
        #    self.special_words), "the value of most_commond_words_num must be larger than the size of special_words"

        """
        vocab_count = counter.most_common(self.top_words - special_words_num)
        vocab = {}
        idx = special_words_num + 1
        for word, _ in vocab_count:
            if word not in self.special_words:
                vocab[word] = idx
                idx += 1
        vocab.update(self.special_words)
        logging.info("store vocabulary with most_common_words file, vocabulary size: " + str(len(vocab)))
        """
        save_obj_pickle(counter, self.vocab_save_path, self.overwrite)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}'.format(p.name))
