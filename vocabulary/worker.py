import zmq
from multiprocessing import Process
from utils.data_util import tokenize
from utils.appmetric_util import with_meter
from utils.retry_util import retry


class WorkerProcess(Process):
    def __init__(self, ip, ventilator_port, port, waiting_time=0.1, threshold=10, name='WorkerProcess'):
        Process.__init__(self)
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.port = port
        self.waiting_time = waiting_time
        self.threshold = threshold
        self.name = name

    @retry(10, zmq.ZMQError, timeout=0.5, name='worker_parser')
    @with_meter('worker_parser', interval=30)
    def _recv_pyboj(self, receiver):
        sentence = receiver.recv_pyobj(zmq.NOBLOCK)
        return sentence

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect("tcp://{}:{}".format(self.ip, self.ventilator_port))

        tokens_words_producer = context.socket(zmq.PUSH)
        tokens_words_producer.connect("tcp://{}:{}".format(self.ip, self.port))
        while True:
            try:
                sentence = self._recv_pyboj(receiver)
            except zmq.ZMQError:
                break
            tokens = tokenize(sentence)
            tokens_words_producer.send_pyobj(tokens)

