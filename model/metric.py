import numpy as np
ignore_label = 0



def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    num = 0.
    for i in range(pred.shape[0]):
        if int(label[i]) != ignore_label:
            num += 1
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return loss / num


def Accuracy(label, pred):
    label = label.T.reshape((-1,))
    pred_indices = np.argmax(pred, axis=1)
    num = 0
    for i in range(pred.shape[0]):
        if int(label[i]) != ignore_label:
            l = sum(map(int, str(int(label[i] - 3)))) if label[i] >= 3 else 0
            p = sum(map(int, str(int(pred_indices[i] - 3)))) if pred_indices[i] >= 3 else 0
            if l == p:
                num += 1
    return float(num) / pred.shape[0] * 100