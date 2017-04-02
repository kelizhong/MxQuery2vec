import zmq
from multiprocessing import Process
import logging
from utils.appmetric_util import AppMetric
from zmq.devices import ProcessDevice


class AksisParseDataBroker(Process):
    def __init__(self, ip, frontend_port=5557, backend_port=5558, name="AksisParsedDataBrokerProcess"):
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
        sender.bind_to_random_port().bind("tcp://{}:{}".format(self.ip, self.backend_port))
        logging.info("start {}, ip:{}, pull port:{}, push port:{}".format(self.name, self.ip, self.frontend_port,
                                                                          self.backend_port))
        metric = AppMetric(name=self.name)
        while True:
            data = receiver.recv_pyobj()
            sender.send_pyobj(data)
            metric.notify(1)
        metric.stop()

class AksisParseDataBroker1(object):
    def __init__(self, ip, frontend_port=5557, backend_port=5558, name="AksisParsedDataBrokerProcess"):
        Process.__init__(self)
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    def run(self):
        dev = ProcessDevice(zmq.STREAMER, zmq.PULL, zmq.PUSH)
        dev.bind_in("tcp://127.0.0.1:%d" % self.frontend_port)
        dev.bind_out("tcp://127.0.0.1:%d" % self.backend_port)
        dev.setsockopt_in(zmq.IDENTITY, b'PULL')
        dev.setsockopt_out(zmq.IDENTITY, b'PUSH')
        dev.start()