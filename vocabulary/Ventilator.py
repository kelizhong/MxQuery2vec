import zmq
import logging
import codecs
from multiprocessing import Process
from worker import WorkerProcess
from collector import  CollectorProcess
from collections import Counter

def sentence_gen(filename):
    """Return each sentence in a line."""
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip().lower()
            if len(line):
                yield line

class VentilatorProcess(Process):
    def __init__(self, corpus_files, ip, port):
        Process.__init__(self)
        self.ip = ip
        self.port = port
        self.corpus_files = [corpus_files] if not isinstance(corpus_files, list) else corpus_files

    def run(self):
        data_context = zmq.Context()
        self.data_socket = data_context.socket(zmq.PUSH)
        self.data_socket.bind("tcp://{}:{}".format("127.0.0.1", self.port))

        logging.info("start sentence producer")
        for filename in self.corpus_files:
            logging.info("Counting words in %s" % filename)
            for sentence in sentence_gen(filename):
                self.data_socket.send_string(sentence)

class Ventilator(object):
    def __init__(self, corpus_files, data_port='5555', status_port='5556'):
        data_context = zmq.Context()
        self.data_socket = data_context.socket(zmq.PUSH)
        self.data_socket.bind("tcp://{}:{}".format("127.0.0.1", data_port))

        status_context = zmq.Context()
        self.status_socket = status_context.socket(zmq.PUB)
        self.statsu_socket.bind("tcp://{}:{}".format("127.0.0.1", status_port))

        self.corpus_files = [corpus_files] if not isinstance(corpus_files, list) else corpus_files

    def produce(self):
        logging.info("start sentence producer")
        for filename in self.corpus_files:
            logging.info("Counting words in %s" % filename)
            for sentence in sentence_gen(filename):
                self.data_socket.send_string(sentence)
        self.statsu_socket.send("DONE")

if __name__ == '__main__':
     v = VentilatorProcess('../data/query2vec/train_corpus/search.keyword.enc', '127.0.0.1', '5555')
     for _ in xrange(8):
        w = WorkerProcess('127.0.0.1', '5555', '5556')
        w.start()
     c = CollectorProcess('127.0.0.1', '5556')
     v.start()
     counter = Counter(c.collect())

     print(v.is_alive())
