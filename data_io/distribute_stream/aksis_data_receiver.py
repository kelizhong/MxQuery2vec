import zmq
from utils.network_util import local_ip
from utils.appmetric_util import with_meter
from utils.log_util import set_up_logger_handler_with_file


class AksisDataReceiver(object):
    """Receiver the data from collector

    Parameters
    ----------
    ip : str
        The ip address string without the port to pass to ``Socket.bind()``.
    port: int
        Port to receive the data from collector
    stop_freq: int
        Frequency to raise the StopIteration error. If the trainer iter receive the StopIteration,
        the trainer will save the checkpoint. If stop_freq < 0, will not raise the StopIteration error
    """
    def __init__(self, ip, port=5556, stop_freq=-1):
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://{}:{}".format(ip, port))
        self.num = 0
        self.stop_freq = stop_freq

    def __iter__(self):
        return self

    @with_meter('aksis_data_receiver', interval=30)
    def next(self):
        if 0 < self.stop_freq < self.num:
            raise StopIteration
        data = self.receiver.recv_pyobj()
        self.num += 1
        return data

    def reset(self):
        self.num = 0

if __name__ == '__main__':
    set_up_logger_handler_with_file('../../configure/logger.conf', 'root')
    print(local_ip())
    r = AksisDataReceiver(local_ip(), port=5558)
    for x in r:
        pass
