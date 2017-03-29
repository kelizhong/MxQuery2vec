import socket
import zmq
from data_io.data_stream.seq2seq_aksis_data_stream import Seq2seqAksisDataStream
from utils.data_util import load_pickle_object
import os
import fnmatch
from multiprocessing import Process
from utils.decorator_util import memoized


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


class Seq2seqDataVentilatorProcess(Process):
    def __init__(self, action_pattern, data_dir, vocabulary_path, batch_size, buckets, num_epoch=65535, ip='127.0.0.1', port='5555', name='VentilatorProcess'):
        Process.__init__(self)
        self.action_pattern = action_pattern
        self.data_dir = data_dir
        self.vocabulary_path = vocabulary_path
        self.batch_size = batch_size
        self.buckets = buckets
        self.num_epoch = num_epoch
        self.ip = ip
        self.port = port
        self.data_stream = self.get_data_stream()

    def run(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        zmq_socket.connect("tcp://{}:{}".format(self.ip, self.port))

        while self.num_epoch > 0:
            for data in self.data_stream:
                print data
                zmq_socket.send_pyobj(data)
            self.num_epoch -= 1
            self.reset()

    def reset(self):
        self.data_stream.reset()

    @property
    @memoized
    def vocabulary(self):
        vocab = load_pickle_object(self.vocabulary_path)
        return vocab

    def get_data_stream(self):
        action_add_files = fnmatch.filter(os.listdir(self.data_dir), self.action_pattern)
        assert len(action_add_files) > 0, "no files are found for action pattern {} in {}".format(self.action_pattern, self.data_dir)
        data_stream = Seq2seqAksisDataStream(action_add_files, self.vocabulary,
                              self.vocabulary, self.buckets, self.batch_size)
        return data_stream

if __name__ == '__main__':
    vocab = load_pickle_object('../../data/vocabulary/vocab.pkl')
    s = Seq2seqDataStream('../../data/query2vec/train_corpus/small.enc', '../../data/query2vec/train_corpus/small.dec', vocab,
                          vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], 2)
    a = Seq2seqDataVentilator(s)
    a.produce()
