import time
from functools import wraps
import types


def retry(tries, delay=1, backoff=1, exception=Exception, name=None, report=lambda *args: None):
    """
    A retry decorator with exponential backoff,
    Retries a function or method if Exception occurred

    Args:
        tries: number of times to retry, set to 0 to disable retry
        delay: initial delay in seconds(can be float, eg 0.01 as 10ms),
            if the first run failed, it would sleep 'delay' second and try again
        backoff: must be greater than 1,
            further failure would sleep delay *= backoff second
    """

    def retrying(func):
        func_name = name or func.__name__

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            _tries = tries(self) if isinstance(tries, types.LambdaType) else tries
            _delay = delay(self) if isinstance(delay, types.LambdaType) else delay
            for i in xrange(_tries):
                try:
                    ret = func(self, *args, **kwargs)
                    return ret
                except exception as e:
                    # retried enough and still fail? raise orignal exception
                    if i == _tries - 1:
                        report("{} failed to retry definitely: {}".format(func_name, e))
                        raise
                    else:
                        report("{} retry {} time, Exception: {}".format(func_name, i, e))
                        time.sleep(_delay)
                        # wait longer after each failure
                        _delay *= backoff

        return wrapper

    return retrying
