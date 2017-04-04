import zmq
from multiprocessing import Process
from utils.data_util import tokenize
from utils.appmetric_util import with_meter
from utils.retry_util import retry
from zmq.decorators import socket
import logging


class WorkerProcess(Process):
    def __init__(self, ip, ventilator_port, port, waiting_time=0.1, threshold=10, name='WorkerProcess'):
        Process.__init__(self)
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.port = port
        self.waiting_time = waiting_time
        self.threshold = threshold
        self.name = name

    @retry(10, exception=zmq.ZMQError, name='worker_parser', report=logging.error)
    @with_meter('worker_parser', interval=30)
    def _on_recv(self, receiver):
        sentence = receiver.recv_string(zmq.NOBLOCK)
        return sentence

    @socket(zmq.PULL)
    @socket(zmq.PUSH)
    def run(self, receiver, sender):
        receiver.connect("tcp://{}:{}".format(self.ip, self.ventilator_port))
        sender.connect("tcp://{}:{}".format(self.ip, self.port))
        while True:
            try:
                sentence = self._on_recv(receiver)
            except zmq.ZMQError as e:
                logging.error(e)
                break
            tokens = tokenize(sentence)
            sender.send_pyobj(tokens)

