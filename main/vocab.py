from vocabulary.ventilator import VentilatorProcess
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from collections import Counter

if __name__ == '__main__':
     v = VentilatorProcess('../data/query2vec/train_corpus/search.keyword.enc', '127.0.0.1', '5555')
     for _ in xrange(8):
        w = WorkerProcess('127.0.0.1', '5555', '5556')
        w.start()
     c = CollectorProcess('127.0.0.1', '5556')
     v.start()
     counter = Counter(c.collect())

     print(v.is_alive())