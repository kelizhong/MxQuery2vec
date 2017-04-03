import logging
import pickle
from multiprocessing import Process

import zmq
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream

from data_io.seq2seq_data_bucket_queue import Seq2seqDataBcuketQueue
from utils.appmetric_util import AppMetric


class AksisDataCollector(Process):
    def __init__(self, ip, buckets, batch_size, frontend_port=5557, backend_port=5558, metric_interval=30,
                 name="AksisDataCollectorProcess"):
        Process.__init__(self)
        self.ip = ip
        self.buckets = buckets
        self.batch_size = batch_size
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.metric_interval = metric_interval
        self.name = name

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender = context.socket(zmq.PUSH)
        sender.bind("tcp://{}:{}".format(self.ip, self.backend_port))
        queue = Seq2seqDataBcuketQueue(self.buckets, self.batch_size)
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        logging.info(
            "start collector {}, ip:{}, frontend port:{}, backend port:{}".format(self.name, self.ip,
                                                                                  self.frontend_port,
                                                                                  self.backend_port))
        ioloop.install()
        loop = ioloop.IOLoop.instance()
        pull_stream = ZMQStream(receiver)

        def _on_recv(msg):
            encoder_sentence_id, decoder_sentence_id, label_id = pickle.loads(msg[0])
            data = queue.put(encoder_sentence_id, decoder_sentence_id, label_id)
            if data:
                sender.send_pyobj(data)
                metric.notify(self.batch_size)

        pull_stream.on_recv(_on_recv)

        loop.start()
