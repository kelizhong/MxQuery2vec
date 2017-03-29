import pickle
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic
def load_vocab(path):
    """
    Load vocab from file, the 0, 1, 2, 3 should be reserved for pad, <unk>, <s>, </s>
    :param path: the vocab
    :param special:
    :return:
    """
    with open(path, 'rb') as f:
        vocab = pickle.load(f)

    return vocab
if __name__ == "__main__":
    vocab = load_vocab('./data/vocabulary/vocab.pkl')
    rev_vocab = MakeRevertVocab(vocab)
    for (k, v) in vocab.items():
        print (k,v)
    print(rev_vocab.get(7, -1))
    print(vocab.get('iphone', -1))