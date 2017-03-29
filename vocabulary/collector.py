import zmq
import time
from appmetrics import metrics
import logging


class CollectorProcess(object):
    def __init__(self, ip, worker_port, waiting_time=0.1, threshold=10):
        self.ip = ip
        self.worker_port = worker_port
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind("tcp://{}:{}".format(self.ip, self.worker_port))
        self.waiting_time = waiting_time
        self.threshold = threshold

    def collect(self):
        num = 0
        retry = 0
        meter = metrics.new_meter("meter_speed")

        while True:
            num += 1
            if num % 100000 == 0:
                print(meter.get())
            try:
                words = self.receiver.recv_pyobj(zmq.NOBLOCK)
                retry = 0
            except zmq.ZMQError:
                if retry > self.threshold:
                    break
                retry += 1
                time.sleep(self.waiting_time)
                logging.info("collector is waiting, has retried {} times".format(retry))
                continue
            meter.notify(1)
            for word in words:
                if len(word):
                    yield word

