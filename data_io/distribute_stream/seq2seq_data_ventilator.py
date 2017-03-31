
import zmq
from data_io.data_stream.seq2seq_aksis_data_stream import Seq2seqAksisDataStream
import os
import fnmatch
from multiprocessing import Process
from utils.decorator_util import memoized
from utils.data_util import load_pickle_object, load_vocabulary_from_pickle
from common.constant import special_words
import logging
from utils.network_util import local_ip


class Seq2seqDataVentilator(object):
    def __init__(self, data_stream, num_epoch=65535, ip=None, port='5555'):
        self.data_stream = data_stream
        self.num_epoch = num_epoch
        self.ip = ip or local_ip()
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUSH)
        self.zmq_socket.bind("tcp://{}:{}".format(self.ip, port))

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
    def __init__(self, action_pattern, data_dir, vocabulary_path, top_words, batch_size, buckets, sample_floor=-1,
                 num_epoch=65535, ip='127.0.0.1', port='5555', name='VentilatorProcess'):
        Process.__init__(self)
        self.action_pattern = action_pattern
        self.data_dir = data_dir
        self.vocabulary_path = vocabulary_path
        self.top_words = top_words
        self.batch_size = batch_size
        self.buckets = buckets
        self.sample_floor = sample_floor
        self.num_epoch = num_epoch
        self.ip = ip
        self.port = port
        self.name = name

    def run(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        zmq_socket.connect("tcp://{}:{}".format(self.ip, self.port))
        logging.info("porcess {} connect {}:{} and start produce data".format(self.name, self.ip, self.port))
        data_stream = self.get_data_stream()
        while self.num_epoch > 0:
            for data in data_stream:
                zmq_socket.send_pyobj(data)
            self.num_epoch -= 1
            data_stream.reset()

    @property
    @memoized
    def vocabulary(self):
        logging.info("loading vocabulary for process {}".format(self.name))
        vocab = load_vocabulary_from_pickle(self.vocabulary_path, top_words=self.top_words, special_words=special_words)
        return vocab

    def get_data_stream(self):
        data_files = fnmatch.filter(os.listdir(self.data_dir), self.action_pattern)
        assert len(data_files) > 0, "no files are found for action pattern {} in {}".format(self.action_pattern,
                                                                                            self.data_dir)
        action_add_files = [os.path.join(self.data_dir, filename) for filename in data_files]
        data_stream = Seq2seqAksisDataStream(action_add_files, self.vocabulary,
                                             self.vocabulary, self.buckets, self.batch_size,
                                             sample_floor=self.sample_floor)
        return data_stream


if __name__ == '__main__':
    vocab = load_pickle_object('../../data/vocabulary/vocab.pkl')
    s = Seq2seqDataStream('../../data/query2vec/train_corpus/small.enc', '../../data/query2vec/train_corpus/small.dec',
                          vocab,
                          vocab, [(3, 10), (3, 20), (5, 20), (7, 30)], 2)
    a = Seq2seqDataVentilator(s)
    a.produce()
