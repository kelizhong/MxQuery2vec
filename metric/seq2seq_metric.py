import numpy as np


class MetricManage(object):
    """metric manage for seq2seq model
    Parameters
    ----------
        ignore_label : int or None
            index of invalid label to ignore when
            counting. usually should be 0. Include
            all entries if None.
    """

    def __init__(self, ignore_label):
        self.ignore_label = ignore_label

    def create_metric(self, metric_name):
        """Create metric by metric name, metric name should be the function name in MetricManage"""
        try:
            metric = getattr(self, metric_name)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement metric `{}`".format(self.__class__.__name__, metric_name))
        return metric

    def perplexity(self, label, pred):
        """Calculate perplexity"""
        label = label.T.reshape((-1,))
        loss = 0.
        num = 0.
        for i in range(pred.shape[0]):
            if int(label[i]) != self.ignore_label:
                num += 1
                loss += -np.log(max(1e-10, pred[i][int(label[i])]))
        return np.exp(loss / num)

    def accuracy(self, label, pred):
        """Calculate predictions accuracy"""
        label = label.T.reshape((-1,))
        pred_indices = np.argmax(pred, axis=1)
        num = 0
        for i in range(pred.shape[0]):
            if int(label[i]) != self.ignore_label:
                l = sum(map(int, str(int(label[i] - 3)))) if label[i] >= 3 else 0
                p = sum(map(int, str(int(pred_indices[i] - 3)))) if pred_indices[i] >= 3 else 0
                if l == p:
                    num += 1
        return float(num) / pred.shape[0] * 100
