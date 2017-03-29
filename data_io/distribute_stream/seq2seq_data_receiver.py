import zmq


class Seq2seqDataReceiver(object):
    def __init__(self, ip_addr, port=5556, send_stop_freq=-1):
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://{}:{}".format(ip_addr, port))
        self.num = 0
        self.send_stop_freq = send_stop_freq

    def consume(self):
        while True:
            data = self.receiver.recv_pyobj()
            print(data)

    def __iter__(self):
        return self

    def next(self):
        if self.send_stop_freq > 0 and self.num > self.send_stop_freq:
            raise StopIteration
        data = self.receiver.recv_pyobj()
        self.num += 1
        return data

    def reset(self):
        self.num = 0


if __name__ == '__main__':
    c = Seq2seqDataReceiver()
    c.consume()
