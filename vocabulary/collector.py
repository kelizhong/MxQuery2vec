import zmq
from utils.appmetric_util import with_meter
from utils.retry_util import retry


class CollectorProcess(object):
    def __init__(self, ip, worker_port, waiting_time=0.3, threshold=10):
        self.ip = ip
        self.worker_port = worker_port
        self.waiting_time = waiting_time
        self.threshold = threshold

    @retry(10, zmq.ZMQError, timeout=0.5, name="vocabulary_collector")
    @with_meter('vocabulary_collector', interval=30)
    def _recv_pyboj(self, receiver):
        words = receiver.recv_pyobj(zmq.NOBLOCK)
        return words

    def collect(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.worker_port))
        while True:
            try:
                words = self._recv_pyboj(receiver)
            except zmq.ZMQError:
                break
            for word in words:
                if len(word):
                    yield word
