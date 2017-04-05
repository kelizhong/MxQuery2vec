import zmq
import logging
from zmq.devices import ProcessDevice


class AksisRawDataBroker(object):
    """Broker between ventilitor process and worker process
    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        frontend_port: int
            Port for the incoming traffic
        backend_port: int
            Port for the outbound traffic
        name: str
            Worker process name
    """
    def __init__(self, ip, frontend_port=5555, backend_port=5556, name="AksisRawDataBroker"):
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    def run(self):
        # start device that will be run in a background Process.
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
