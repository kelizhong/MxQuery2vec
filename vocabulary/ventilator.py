import zmq
import logging
from multiprocessing import Process
from worker import WorkerProcess
from collector import CollectorProcess
from collections import Counter
from utils.data_util import sentence_gen


class VentilatorProcess(Process):
    def __init__(self, corpus_files, ip, port, sentence_gen=sentence_gen, name='VentilatorProcess'):
        Process.__init__(self)
        self.ip = ip
        self.port = port
        self.corpus_files = [corpus_files] if not isinstance(corpus_files, list) else corpus_files
        self.sentence_gen = sentence_gen

    def run(self):
        data_context = zmq.Context()
        data_socket = data_context.socket(zmq.PUSH)
        data_socket.bind("tcp://{}:{}".format(self.ip, self.port))

        logging.info("start sentence producer")
        for filename in self.corpus_files:
            logging.info("Counting words in %s" % filename)
            for sentence in self.sentence_gen(filename):
                data_socket.send_string(sentence)


if __name__ == '__main__':
     v = VentilatorProcess('../data/query2vec/train_corpus/search.keyword.enc', '127.0.0.1', '5555')
     for _ in xrange(8):
        w = WorkerProcess('127.0.0.1', '5555', '5556')
        w.start()
     c = CollectorProcess('127.0.0.1', '5556')
     v.start()
     counter = Counter(c.collect())

     print(v.is_alive())
