import zmq
import time
import logging
from utils.appmetric_util import AppMetric


class CollectorProcess(object):
    def __init__(self, ip, worker_port, waiting_time=0.3, threshold=10):
        self.ip = ip
        self.worker_port = worker_port
        self.waiting_time = waiting_time
        self.threshold = threshold

    def collect(self):
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.worker_port))
        retry = 0
        metric = AppMetric()

        while True:

            try:
                words = receiver.recv_pyobj(zmq.NOBLOCK)
                retry = 0
            except zmq.ZMQError:
                if retry > self.threshold:
                    metric.stop()
                    break
                retry += 1
                time.sleep(self.waiting_time)
                logging.info("collector is waiting, has retried {} times".format(retry))
                continue
            metric.notify(1)
            for word in words:
                if len(word):
                    yield word

