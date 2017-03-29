from data_io.distribute_stream.seq2seq_data_ventilator import Seq2seqDataVentilatorProcess
from data_io.distribute_stream.seq2seq_data_broker import Seq2seqDataBroker


class Seq2seqDataManager(object):
    def __init__(self, data_dir, vocabulary_path, action_patterns, batch_size, buckets, ip='127.0.0.1', pull_port='5555', push_port='5556',
                 num_epoch=65535):
        self.data_dir = data_dir
        self.vocabulary_path = vocabulary_path
        self.ip = ip
        self.pull_port = pull_port
        self.push_port = push_port
        self.action_patterns = action_patterns
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.buckets = buckets

    def start_data_stream_process(self):
        for action_pattern in self.action_patterns:
            p = Seq2seqDataVentilatorProcess(action_pattern, self.data_dir, self.vocabulary_path, self.batch_size, self.buckets,
                                             num_epoch=self.num_epoch, ip=self.ip, port=self.pull_port,
                                             name='VentilatorProcess')
            p.start()

    def start_data_broker(self):
        broker = Seq2seqDataBroker(self.ip, self.pull_port, self.push_port)
        broker.start()

    def start_all(self):
        self.start_data_stream_process()
        self.start_data_broker()
