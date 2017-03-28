from multiprocessing import Pool, cpu_count
import multiprocessing
import os, time, random
import copy_reg
import types
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def long_time_task(name):
    print 'Run task %s (%s)...' % (name, os.getpid())
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print 'Task %s runs %0.2f seconds.' % (name, (end - start))

class ProcessManager(object):
    def __init__(self):
        self.pool = multiprocessing.Pool()
        m = multiprocessing.Manager()
        self.sentence_queue = m.Queue()
        self.token_words_queue = m.Queue()
        self.start()

    def long_time_task(self, name):
        print 'Run task %s (%s)...' % (name, os.getpid())
        start = time.time()
        time.sleep(random.random() * 3)
        end = time.time()
        print 'Task %s runs %0.2f seconds.' % (name, (end - start))


    def start(self):
        for i in range(5):
            self.pool.apply_async(self.long_time_task, args=(i,))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        #print self.__dict__
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        #print state
        self.__dict__.update(state)


if __name__=='__main__':
    print 'Parent process %s.' % os.getpid()
    p = ProcessManager()
    #p.start()
    p.pool.close()
    p.pool.join()
    print 'All subprocesses done.'