# coding=utf-8
import logging
from utils.data_util import sentence_gen
import multiprocessing
from nltk.tokenize import word_tokenize
from multiprocessing import Process, Queue


class TaskProcessManager(object):
    def __init__(self, corpus_files, tokenizer_num=1):
        self.corpus_files = corpus_files
        self.corpus_files = [corpus_files] if not isinstance(corpus_files, list) else corpus_files
        self.tokenizer_num = tokenizer_num
        self._init_process()

    def _init_process(self):
        self.m = multiprocessing.Manager()

        self.producer_event = self.m.Event()
        self.sentence_queue = self.m.Queue()

        self.tokenizer_event_pool = []
        self.tokenizer_pool = []
        self.token_words_queue = self.m.Queue()

        self.producer = Process(target=self.produce_sentence)

        for i in xrange(self.tokenizer_num):
            event = self.m.Event()
            self.tokenizer_event_pool.append(event)
            p = Process(target=self.tokenize_sentence, args=(i, event, ))
            self.tokenizer_pool.append(p)

    def produce_sentence(self):
        logging.info("start sentence producer")
        for filename in self.corpus_files:
            logging.info("Counting words in %s" % filename)
            for sentence in sentence_gen(filename):
                self.sentence_queue.put(sentence)
        self.producer_event.set()

    def tokenize_sentence(self, rank, event):
        logging.info("start sentence tokenizer {}". format(rank))

        while self.has_element_in_sentence_queue():
            try:
                sentence = self.sentence_queue.get(timeout=1)
            except Queue.Empty:
                logging.warning("sentence_queue is empty")
                continue
            tokens = word_tokenize(sentence)
            self.token_words_queue.put(tokens)
        event.set()

    def has_element_in_sentence_queue(self):
        return not self.sentence_queue.empty() or not self.is_done_producer_process()

    def has_element_in_token_words_queue(self):
        return not self.is_all_done_tokenizer_process() or not self.token_words_queue.empty()

    def is_all_done_tokenizer_process(self):
        return all(event.is_set() for event in self.tokenizer_event_pool)

    def is_done_producer_process(self):
        return self.producer_event.is_set()

    def word_gen(self):
        while self.has_element_in_token_words_queue():
            try:
                tokens = self.token_words_queue.get(timeout=1)
            except Queue.Empty:
                logging.warning("token_words_queue is empty")
                continue
            for word in tokens:
                if len(word):
                    yield word

    def start(self):
        self.producer.start()
        for p in self.tokenizer_pool:
            p.start()
