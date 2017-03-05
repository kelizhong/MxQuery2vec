import pickle
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
    print(vocab.get('know', -1))