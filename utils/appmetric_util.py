from appmetrics import metrics
import logging
from appmetrics import reporter
import functools


def with_meter(name, value=1, interval=-1):
    """
    Call-counting decorator: each time the wrapped function is called
    the named meter is incremented by one.
    metric_args and metric_kwargs are passed to new_meter()
    """

    try:
        mmetric = AppMetric(name=name, interval=interval)

    except metrics.DuplicateMetricError as e:
        mmetric = AppMetric(metric=metrics.metric(name), interval=interval)

    def wrapper(f):

        @functools.wraps(f)
        def fun(*args, **kwargs):
            res = f(*args, **kwargs)

            mmetric.notify(value)
            return res

        return fun

    return wrapper


class AppMetric(object):
    def __init__(self, metric=None, name='metric', interval=-1):
        self.metric = metric or metrics.new_meter(name)

        self.interval = interval
        if interval > 0:
            reporter.register(self.log_report, reporter.fixed_interval_scheduler(interval))

    @staticmethod
    def log_report(metrics):
        logging.info(metrics)

    def notify(self, value):
        self.metric.notify(value)

