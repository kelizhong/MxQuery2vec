import logging
import time


class Speedometer(object):
    """Calculate and log training speed periodically.

    Parameters
    ----------
        batch_size: int
            Batch_size of data
        rank: int
            The rank of worker node, which is in [0, kv.get_num_workers())
        frequent: int
            How many batches between calculations.
            Defaults to calculating & logging every 50 batches.
    """
    def __init__(self, batch_size, rank=0, frequent=50):
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
                        logging.info('Worker[%d] Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                     self.rank, param.epoch, count, speed, name, value)
                else:
                    logging.info("Worker[%d] Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 self.rank, param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
