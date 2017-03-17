import numpy as np
from operator import itemgetter
import mxnet as mx
class NceAuc(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAuc, self).__init__('nce-auc')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key = itemgetter(1), reverse = True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, b in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1


class MetricManage(object):
    def __init__(self):
        pass

    def create_metric(self, metric_name):
        try:
            metric = getattr(self, metric_name)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement metric `{}`".format(self.__class__.__name__, metric_name))
        return metric

    @staticmethod
    def nce_auc(labels, preds):
        label_weight = labels[1].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key = itemgetter(1), reverse = True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, b in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        return z
