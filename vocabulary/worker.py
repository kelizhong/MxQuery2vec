import zmq
import logging
from nltk.tokenize import word_tokenize
from multiprocessing import Process
import time

class Worker(object):
    def __init__(self, data_port='5555', ventilator_status_port='5556', status='5558'):
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://{}:{}".format("127.0.0.1", data_port))

        self.tokens_words_producer = context.socket(zmq.PUSH)
        self.tokens_words_producer.connect("tcp://{}:{}".format("127.0.0.1", '5556'))

    def consume(self):
        while True:
            sentence = self.receiver.recv_string()
            tokens = word_tokenize(sentence)
            self.tokens_words_producer.send_pyobj(tokens)

class WorkerProcess(Process):
    def __init__(self, ip, ventilator_port, port, waiting_time=0.1, threshold=10):
        Process.__init__(self)
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.port = port
        self.waiting_time = waiting_time
        self.threshold = threshold



    def run(self):
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://{}:{}".format("127.0.0.1", self.ventilator_port))

        context = zmq.Context()
        self.tokens_words_producer = context.socket(zmq.PUSH)
        self.tokens_words_producer.connect("tcp://{}:{}".format("127.0.0.1", self.port))

        retry = 0
        while True:
            try:
                sentence = self.receiver.recv_string(zmq.NOBLOCK)
                retry = 0
            except zmq.ZMQError:
                if retry > self.threshold:
                    break
                retry += 1
                logging.info("working is waiting, has retried {} times".format(retry))
                time.sleep(self.waiting_time)
                continue
            tokens = word_tokenize(sentence)
            self.tokens_words_producer.send_pyobj(tokens)

