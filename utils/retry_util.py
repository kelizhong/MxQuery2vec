import decorator
import time
import logging


def retry(times, *exception_types, **kwargs):
    timeout = kwargs.get('timeout', 0.0)  # seconds
    name = kwargs.get('name')  # name

    @decorator.decorator
    def retrying(func, *func_args, **func_kwargs):
        func_name = name or func.__name__
        for i in xrange(times):
            try:
                logging.warn("{} is waiting, has retried {} times".format(func_name, i))
                return func(*func_args, **func_kwargs)
            except exception_types:
                if timeout is not None:
                    time.sleep(timeout)
        raise exception_types
    return retrying
