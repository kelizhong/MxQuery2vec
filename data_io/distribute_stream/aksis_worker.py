import zmq
from data_io.data_stream.seq2seq_aksis_data_stream import Seq2seqAksisDataStream
import threading
from multiprocessing import Process
from utils.decorator_util import memoized
from utils.data_util import  load_vocabulary_from_pickle
from common.constant import special_words
import logging
from Queue import Queue


class AksisWorker(Process):
    def __init__(self, ip, frontend_port=5556, backend_port=5557, name="AksisWorkerProcess"):
        Process.__init__(self)
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name
        self.queue = Queue()

    def add_to_queue(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        while True:
            self.queue.put(receiver.recv_pyobj())

    def start_receiver_thread(self):
        t = threading.Thread(target=self.add_to_queue())
        t.setDaemon(True)
        t.start()

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender = context.socket(zmq.PULL)
        sender.connect("tcp://{}:{}".format(self.ip, self.backend_port))
        self.start_receiver_thread()
        data_stream = self.get_data_stream()
        while True:
            for data in data_stream:
                sender.send_pyobj(data)



    @property
    @memoized
    def vocabulary(self):
        logging.info("loading vocabulary for process {}".format(self.name))
        vocab = load_vocabulary_from_pickle(self.vocabulary_path, top_words=self.top_words, special_words=special_words)
        return vocab

    def get_data_stream(self):

        data_stream = Seq2seqAksisDataStream(self.queue, self.vocabulary,
                                             self.vocabulary, self.buckets, self.batch_size,
                                             sample_floor=self.sample_floor)
        return data_stream