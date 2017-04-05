import zmq
import os
import fnmatch
from multiprocessing import Process
import logging
from utils.data_util import query_title_score_generator_from_aksis_data
import random
from zmq.decorators import socket
from utils.appmetric_util import AppMetric


class AksisDataVentilatorProcess(Process):
    """Process to read the corpus data
    Parameters
    ----------
        data_dir : str
            Data_dir for the aksis corpus data
        file_pattern: tuple
            File pattern use to distinguish different corpus, every file pattern will start a ventilitor process.
            File pattern is tuple type(file pattern, dropout). Dropout is the probability to ignore the data.
            If dropout < 0, all the data will be accepted to be trained
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        port: int
            Port to produce the raw data
        num_epoch: int
            end epoch of producing the data
        name: str
            process name
    """
    def __init__(self, file_pattern, data_dir,
                 num_epoch=65535, dropout=-1, ip='127.0.0.1', port='5555', metric_interval=30, name='VentilatorProcess'):
        Process.__init__(self)
        self.file_pattern = file_pattern
        self.data_dir = data_dir
        self.num_epoch = num_epoch
        self.dropout = float(dropout)
        self.ip = ip
        self.port = port
        self.metric_interval = metric_interval
        self.name = name

    @socket(zmq.PUSH)
    def run(self, sender):
        sender.connect("tcp://{}:{}".format(self.ip, self.port))
        logging.info("process {} connect {}:{} and start produce data".format(self.name, self.ip, self.port))
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        data_stream = self.get_data_stream()
        for i in xrange(self.num_epoch):
            for data in data_stream:
                sender.send_pyobj(data)
                metric.notify(1)
            data_stream = self.get_data_stream()
            logging.info("process {} finish {} epoch".format(self.name, i))

    def get_data_stream(self):
        data_files = fnmatch.filter(os.listdir(self.data_dir), self.file_pattern)
        assert len(data_files) > 0, "no files are found for action pattern {} in {}".format(self.file_pattern,
                                                                                            self.data_dir)
        action_add_files = [os.path.join(self.data_dir, filename) for filename in data_files]

        for query, title, score in query_title_score_generator_from_aksis_data(action_add_files):
            if self.is_hit(score):
                yield query, title

    def is_hit(self, score):
        """sample function to decide whether the data should be trained, not sample if dropout less than 0"""
        return self.dropout < 0 or float(score) > random.uniform(self.dropout, 1)

