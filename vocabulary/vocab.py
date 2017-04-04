# coding=utf-8
import logging
from collections import Counter
from utils.pickle_util import save_obj_pickle
from vocabulary.ventilator import VentilatorProcess
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from utils.data_util import sentence_gen


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

    def __init__(self, corpus_files, vocab_save_path, sentence_gen=sentence_gen, workers_num=1, top_words=100000,
                 ip='127.0.0.1', ventilator_port='5555', collector_port='5556',
                  overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_save_path = vocab_save_path
        self.sentence_gen = sentence_gen
        self.workers_num = workers_num
        self.top_words = top_words
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
        self.overwrite = overwrite

    def create_dictionary(self):
        process_pool = []
        v = VentilatorProcess(self.corpus_files, self.ip, self.ventilator_port, sentence_gen=self.sentence_gen)
        v.start()
        process_pool.append(v)
        for i in xrange(self.workers_num):
            w = WorkerProcess(self.ip, self.ventilator_port, self.collector_port, name='WorkerProcess_{}'.format(i))
            w.start()
            process_pool.append(w)
        c = CollectorProcess(self.ip, self.collector_port)
        counter = Counter(c.collect())
        self._terminate_process(process_pool)
        logging.info("Finish counting. %d unique words, a total of %d words in all files."
                     % (len(counter), sum(counter.values())))

        counter = counter.most_common(self.top_words)
        logging.info("store vocabulary with most_common_words file, vocabulary size: " + str(len(counter)))
        save_obj_pickle(counter, self.vocab_save_path, self.overwrite)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}'.format(p.name))
