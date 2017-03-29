import zmq
from multiprocessing import Process


class Seq2seqDataBroker(Process):
    def __init__(self, ip, pull_port=5555, push_port=5556):
        self.ip = ip
        self.pull_port = pull_port
        self.push_port = push_port

    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.pull_port))
        sender = context.socket(zmq.PUSH)
        sender.bind("tcp://{}:{}".format(self.ip, self.push_port))
        while True:
            data = self.receiver.recv_pyobj()
            self.sender.send_pyobj(data)

