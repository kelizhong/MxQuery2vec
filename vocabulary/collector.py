import zmq
from utils.appmetric_util import with_meter
from utils.retry_util import retry
import logging


class CollectorProcess(object):
    def __init__(self, ip, worker_port, waiting_time=0.3, tries=20):
        self.ip = ip
        self.worker_port = worker_port
        self.waiting_time = waiting_time
        self.tries = tries

    @retry(lambda x: x.tries, exception=zmq.ZMQError, name="vocabulary_collector", report=logging.error)
    @with_meter('vocabulary_collector', interval=30)
    def _on_recv(self, receiver):
        words = receiver.recv_pyobj(zmq.NOBLOCK)
        return words

    def collect(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.worker_port))
        while True:
            try:
                words = self._on_recv(receiver)
            except zmq.ZMQError as e:
                logging.error(e)
                break
            for word in words:
                if len(word):
                    yield word
