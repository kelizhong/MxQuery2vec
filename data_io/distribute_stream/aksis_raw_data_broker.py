import zmq
import logging
from zmq.devices import ProcessDevice


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
        logging.info(
            "start broker {}, ip:{}, frontend port:{}, backend port:{}".format(self.name, self.ip, self.frontend_port,
                                                                               self.backend_port))

    def start(self):
        self.run()
