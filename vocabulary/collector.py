import zmq
from nltk.tokenize import word_tokenize
from collections import Counter
from multiprocessing import Process
import time
from appmetrics import metrics
import logging

class Collector(object):
    def __init__(self, port='5556'):
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind("tcp://{}:{}".format("127.0.0.1", port))

    def collect(self):
        num = 0
        while True:
            num += 1
            if num % 10000 == 0:
                print(num)
            sentence = self.receiver.recv_pyobj()
            for word in sentence:
                if len(word):
                    yield word


class CollectorProcess(object):
    def __init__(self, ip, worker_port, waiting_time=0.1, threshold=10):
        self.ip = ip
        self.worker_port = worker_port
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind("tcp://{}:{}".format("127.0.0.1", self.worker_port))
        self.waiting_time = waiting_time
        self.threshold = threshold

    def collect(self):
        num = 0
        retry = 0
        meter = metrics.new_meter("meter_test")

        while True:
            num += 1
            if num % 10000 == 0:
                print(meter.get())
            try:
                words = self.receiver.recv_pyobj(zmq.NOBLOCK)
                retry = 0
            except zmq.ZMQError:
                if retry > self.threshold:
                    break
                retry += 1
                time.sleep(self.waiting_time)
                logging.info("collector is waiting, has retried {} times".format(retry))
                continue
            meter.notify(1)
            for word in words:
                if len(word):
                    yield word



if __name__ == '__main__':
    c = Collector()
    global_counter = Counter()
    counter = Counter(c.collect())
    print(len(counter))
