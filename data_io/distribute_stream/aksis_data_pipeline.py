from utils.network_util import local_ip
from aksis_raw_data_broker import AksisRawDataBroker
from aksis_parser_worker import AksisParserWorker
from aksis_ventilator import AksisDataVentilatorProcess
from aksis_data_collector import AksisDataCollector


class AksisDataPipeline(object):
    def __init__(self, data_dir, vocabulary_path, top_words, action_patterns, batch_size, buckets, worker_num=10,
                  ip=None,
                 raw_data_frontend_port='5555', raw_data_backend_port='5556',
                 collector_fronted_port='5557', collector_backend_port='5558',
                 num_epoch=65535):
        self.data_dir = data_dir
        self.vocabulary_path = vocabulary_path
        self.top_words = top_words
        self.ip = ip or local_ip()
        self.raw_data_frontend_port = raw_data_frontend_port
        self.raw_data_backend_port = raw_data_backend_port
        self.collector_fronted_port = collector_fronted_port
        self.collector_backend_port = collector_backend_port
        self.action_patterns = action_patterns
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.buckets = buckets
        self.worker_num = worker_num

    def start_collector_process(self, join=False):
        c = AksisDataCollector(self.ip, self.buckets, self.batch_size,frontend_port=self.collector_fronted_port, backend_port=self.collector_backend_port)
        c.start()
        if join:
            c.join()

    def start_parser_worker_process(self):
        for i in xrange(self.worker_num):
            w = AksisParserWorker(self.ip, self.vocabulary_path, self.top_words, frontend_port=self.raw_data_backend_port, backend_port=self.collector_fronted_port, name="aksis_parser_worker_{}".format(i))
            w.start()

    def start_data_ventilitor_process(self):
        for i, (action_pattern, dropout) in enumerate(self.action_patterns):
            v = AksisDataVentilatorProcess(action_pattern, self.data_dir, dropout=dropout, ip=self.ip, port=self.raw_data_frontend_port, name="aksis_ventilitor_{}".format(i))
            v.start()

    def start_data_broker(self):
        raw_data_broker = AksisRawDataBroker(self.ip, self.raw_data_frontend_port, self.raw_data_backend_port)
        raw_data_broker.start()

    def start_all(self):
        self.start_data_broker()
        self.start_data_ventilitor_process()
        self.start_parser_worker_process()
        self.start_collector_process(join=True)
