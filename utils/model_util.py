import os
from utils.file_util import ensure_dir_exists
import logging
import mxnet as mx
import time


def load_model(model_prefix, rank=0, load_epoch=None):
    if load_epoch is None:
        return None, None, None
    assert model_prefix is not None
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, load_epoch)
    return sym, arg_params, aux_params


def save_model(model_prefix, rank=0, period=10):
    if model_prefix is None:
        return None
    ensure_dir_exists(model_prefix, is_dir=False)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (
        model_prefix, rank), period)


def init_log(log_level, log_path):
    head = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(format=head,
                        level=log_level,
                        datefmt='%H:%M:%S')
    if log_level is not None and log_path is not None:
        ensure_dir_exists(log_path)
        file_handler = logging.FileHandler(os.path.join(log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
        file_handler.setFormatter(logging.Formatter(head))
        logging.root.addHandler(file_handler)


class Speedometer(object):
    """Calculate and log training speed periodically.

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.
    """
    def __init__(self, batch_size, rank=0,frequent=50):
        self.batch_size = batch_size
        self.rank = rank
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    param.eval_metric.reset()
                    for name, value in name_value:
                        logging.info('Node[%d] Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                     self.rank, param.epoch, count, speed, name, value)
                else:
                    logging.info("Node[%d] Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 self.rank, param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()