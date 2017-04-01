from appmetrics import metrics
import threading
import logging


class AppMetric(object):
    def __init__(self, name='metric', interval=10):
        self.meter = metrics.new_meter(name)
        self.interval = interval
        self.log_metric()

    def log_metric(self):
        logging.info(self.meter.get())
        threading.Timer(self.interval, self.log_metric).start()

    def notify(self, value):
        self.meter.notify(value)

