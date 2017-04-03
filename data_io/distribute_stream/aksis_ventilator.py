import zmq
import os
import fnmatch
from multiprocessing import Process
import logging
from utils.data_util import query_title_score_generator_from_aksis_data
import random


class AksisDataVentilatorProcess(Process):
    def __init__(self, action_pattern, data_dir,
                 num_epoch=65535, dropout=-1, ip='127.0.0.1', port='5555', name='VentilatorProcess'):
        Process.__init__(self)
        self.action_pattern = action_pattern
        self.data_dir = data_dir
        self.num_epoch = num_epoch
        self.dropout = float(dropout)
        self.ip = ip
        self.port = port
        self.name = name

    def run(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        zmq_socket.connect("tcp://{}:{}".format(self.ip, self.port))
        logging.info("process {} connect {}:{} and start produce data".format(self.name, self.ip, self.port))

        data_stream = self.get_data_stream()
        while self.num_epoch > 0:
            for data in data_stream:
                zmq_socket.send_pyobj(data)
            self.num_epoch -= 1
            data_stream = self.get_data_stream()

    def get_data_stream(self):
        data_files = fnmatch.filter(os.listdir(self.data_dir), self.action_pattern)
        assert len(data_files) > 0, "no files are found for action pattern {} in {}".format(self.action_pattern,
                                                                                            self.data_dir)
        action_add_files = [os.path.join(self.data_dir, filename) for filename in data_files]

        for query, title, score in query_title_score_generator_from_aksis_data(action_add_files):
            if self.is_hit(score):
                yield query, title

    def is_hit(self, score):
        """sample function to decide whether the data should be trained, not sample if floor less than 0"""
        return self.dropout < 0 or float(score) > random.uniform(self.dropout, 1)

