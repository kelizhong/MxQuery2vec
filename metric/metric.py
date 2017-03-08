import numpy as np
from utils.data_utils import read_data, sentence2id, load_vocab
vocab = load_vocab('./data/vocabulary/vocab.pkl')
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic
vocab = MakeRevertVocab(vocab)

# Evaluation
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    mask_count = 0
    str = ''
    for i in range(pred.shape[0]):
        if int(label[i]) == 0:
            mask_count += 1
            continue

        str = str + " " + vocab.get(int(label[i]), 'unk')
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    print(str)
    return np.exp(loss / (label.size - mask_count))

