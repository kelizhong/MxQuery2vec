import socket
import zmq
from data_io.data_stream.seq2seq_data_stream import Seq2seqDataStream
from utils.data_util import load_pickle_object

def local_ip():
    return socket.gethostbyname(socket.gethostname())


class Seq2seqDataVentilator(object):
    def __init__(self, data_stream, num_epoch=65535, ip_addr=None, port='5555'):
        self.data_stream = data_stream
        self.num_epoch = num_epoch
        #self.ip_addr = local_ip() if ip_addr is None else ip_addr
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUSH)
        self.zmq_socket.bind("tcp://{}:{}".format("127.0.0.1", port))

    def produce(self):
        while self.num_epoch > 0:
            for data in self.data_stream:
                print data
                self.zmq_socket.send_pyobj(data)
            self.num_epoch -= 1
            self.reset()

    def reset(self):
        self.data_stream.reset()

if __name__ == '__main__':
    vocab = load_pickle_object('../../data/vocabulary/vocab.pkl')
    s = Seq2seqDataStream('../../data/query2vec/train_corpus/small.enc', '../../data/query2vec/train_corpus/small.dec', vocab,
                          vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], 2)
    a = Seq2seqDataVentilator(s)
    a.produce()
