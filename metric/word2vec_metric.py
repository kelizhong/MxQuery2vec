# coding=utf-8
from operator import itemgetter
import mxnet as mx

# TODO add metric manage like seq2seq_metric. Shape problem happen when use metric manage like seq2seq_metric


class NceAuc(mx.metric.EvalMetric):
    """Calculate noise-contrastive estimation auc metric
    This metric copied from MXNET Package.
    Currently have no time to dive deep nce loss, will add
    more code annotation after the investigation
    """
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
