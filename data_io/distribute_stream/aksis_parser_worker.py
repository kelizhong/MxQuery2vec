import zmq
from multiprocessing import Process
from utils.decorator_util import memoized
from utils.data_util import load_vocabulary_from_pickle
from common.constant import special_words
import logging
from utils.data_util import convert_data_to_id
from utils.data_util import tokenize
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream
import pickle


class AksisParserWorker(Process):
    def __init__(self, ip, vocabulary_path, top_words, frontend_port=5556, backend_port=5557,
                 name="AksisWorkerProcess"):
        Process.__init__(self)
        self.ip = ip
        self.vocabulary_path = vocabulary_path
        self.top_words = top_words
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender = context.socket(zmq.PUSH)
        sender.connect("tcp://{}:{}".format(self.ip, self.backend_port))
        logging.info("process {} connect {}:{} and start parse data".format(self.name, self.ip, self.frontend_port))
        ioloop.install()
        loop = ioloop.IOLoop.instance()
        pull_stream = ZMQStream(receiver, loop)

        def _on_recv(msg):
            query, title = pickle.loads(msg[0])
            query_words = tokenize(query)
            title_words = tokenize(title)
            data_id = convert_data_to_id(query_words, title_words,
                                         self.vocabulary, self.vocabulary)
            sender.send_pyobj(data_id)

        pull_stream.on_recv(_on_recv)
        loop.start()

    @property
    @memoized
    def vocabulary(self):
        logging.info("loading vocabulary for process {}".format(self.name))
        vocab = load_vocabulary_from_pickle(self.vocabulary_path, top_words=self.top_words, special_words=special_words)
        return vocab
