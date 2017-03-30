import zmq
from multiprocessing import Process
import logging


class Seq2seqDataBroker(Process):
    def __init__(self, ip, pull_port=5555, push_port=5556, name="Seq2seqDataBrokerProcess"):
        Process.__init__(self)
        self.ip = ip
        self.pull_port = pull_port
        self.push_port = push_port
        self.name = name

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.pull_port))
        sender = context.socket(zmq.PUSH)
        sender.bind("tcp://{}:{}".format(self.ip, self.push_port))
        logging.info("start {}, ip:{}, pull port:{}, push port:{}".format(self.ip, self.pull_port, self.push_port))
        while True:
            data = receiver.recv_pyobj()
            sender.send_pyobj(data)

