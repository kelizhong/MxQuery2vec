from appmetrics import metrics
import threading
import logging
from utils.app_scheduler_util import RepeatedTimer


class AppMetric(object):
    def __init__(self, name='metric', interval=10):
        self.meter = metrics.new_meter(name)
        self.interval = interval
        self.name = name
        self.rt = RepeatedTimer(interval, self.log_metric)

    def log_metric(self):
        logging.info("{}:{}".format(self.name, self.meter.get()))

    def notify(self, value):
        self.meter.notify(value)

    def stop(self):
        self.rt.stop()

