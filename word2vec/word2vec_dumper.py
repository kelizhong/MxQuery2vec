from utils.data_util import load_pickle_object
from utils.decorator_util import memoized
from utils.model_util import load_model
from utils.pickle_util import save_obj_pickle


class W2vDumper(object):
    """
    Dump the word2vec from pretrain model into pickle format.
    The object of the word2vec is a dict(key: word, value: vector)
    Parameters
    ----------
    w2v_model_path: str
        the model file path for pretrain word2vec model
    vocab_path: str
        vocabulary file related to the pretrain word2vec model
    w2v_save_path: str
        word2vec file path where the vocabulary will be created
    rank: int
        the rank of worker node
    load_epoch: int
        Epoch number of model we would like to load.
    embedding_weight_name: str
        the name of embedding weight, this name should be same with the embeding name in pretrain model
    """

    def __init__(self, w2v_model_path, vocab_path, w2v_save_path, rank=0, load_epoch=376,
                 embedding_weight_name='embed_weight'):
        self.w2v_model_path = w2v_model_path
        self.vocab_path = vocab_path
        self.w2v_save_path = w2v_save_path
        self.rank = rank
        self.load_epoch = load_epoch
        self.embedding_weight_name = embedding_weight_name

    @property
    @memoized
    def _vocab(self):
        """load the vocabulary which is vocabulary of corpus for training the word2vec model"""
        return load_pickle_object(self.vocab_path)

    @property
    @memoized
    def _embedding(self):
        """load model and extract the embedding weight"""
        _, arg_params, _ = load_model(self.w2v_model_path, self.rank, self.load_epoch)
        assert self.embedding_weight_name in arg_params, "{} parameter not in the w2v query2vec, " \
                                                         "please check the embedding weight name".format(
            self.embedding_weight_name)

        return arg_params.get(self.embedding_weight_name)

    def dumper(self):
        """dump the embedding weight(word2vec) into pickle"""
        vocab = self._vocab
        assert vocab is not None and len(vocab) > 0, "vocabulary can not be empty"
        embed = self._embedding
        assert embed is not None, "Failed to load the embedding"
        embed_np = embed.asnumpy()
        word_num, embed_size = embed_np.shape
        w2v = dict()
        for key, value in vocab.iteritems():
            assert word_num - 1 > value, "vocab index {} out of bound({})".format(value, word_num)
            w2v.setdefault(key, embed_np[value])
        save_obj_pickle(w2v, self.w2v_save_path, overwrite=True)
        return w2v


if __name__ == '__main__':
    W2vDumper('../data/word2vec/model/word2vec', '../data/vocabulary/w2v_vocab.pkl',
              '../data/word2vec/w2v.pkl').dumper()
