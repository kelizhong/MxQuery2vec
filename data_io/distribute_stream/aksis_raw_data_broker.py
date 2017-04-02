import zmq
from multiprocessing import Process
import logging
from utils.appmetric_util import AppMetric
from zmq.devices import ProcessDevice
import time


class AksisRawDataBroker1(Process):
    def __init__(self, ip, frontend_port=5555, backend_port=5556, name="AksisRawDataBrokerProcess"):
        Process.__init__(self)
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender = context.socket(zmq.PUSH)
        sender.bind("tcp://{}:{}".format(self.ip, self.backend_port))
        logging.info("start {}, ip:{}, pull port:{}, push port:{}".format(self.name, self.ip, self.frontend_port,
                                                                          self.backend_port))
        metric = AppMetric()
        while True:
            data = receiver.recv_pyobj()
            sender.send_pyobj(data)
            metric.notify(1)


class AksisRawDataBroker(object):
    def __init__(self, ip, frontend_port=5555, backend_port=5556, name="AksisRawDataBroker"):
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    def run(self):
        dev = ProcessDevice(zmq.STREAMER, zmq.PULL, zmq.PUSH)
        dev.bind_in("tcp://{}:{}".format(self.ip, self.frontend_port))
        dev.bind_out("tcp://{}:{}".format(self.ip, self.backend_port))
        dev.setsockopt_in(zmq.IDENTITY, b'PULL')
        dev.setsockopt_out(zmq.IDENTITY, b'PUSH')
        dev.start()

    def start(self):
        self.run()