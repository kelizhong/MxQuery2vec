import zmq
import logging
from nltk.tokenize import word_tokenize
from multiprocessing import Process
import time


class WorkerProcess(Process):
    def __init__(self, ip, ventilator_port, port, waiting_time=0.1, threshold=10, name='WorkerProcess'):
        Process.__init__(self)
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.port = port
        self.waiting_time = waiting_time
        self.threshold = threshold

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect("tcp://{}:{}".format(self.ip, self.ventilator_port))

        tokens_words_producer = context.socket(zmq.PUSH)
        tokens_words_producer.connect("tcp://{}:{}".format(self.ip, self.port))

        retry = 0
        while True:
            try:
                sentence = receiver.recv_string(zmq.NOBLOCK)
                retry = 0
            except zmq.ZMQError:
                if retry > self.threshold:
                    break
                retry += 1
                logging.info("working is waiting, has retried {} times".format(retry))
                time.sleep(self.waiting_time)
                continue
            tokens = word_tokenize(sentence)
            tokens_words_producer.send_pyobj(tokens)

