from utils.data_util import load_vocab
from utils.decorator_util import memoized
from utils.model_util import load_model


class W2vDumper(object):
    def __init__(self, model_path, vocab_path, embedding_save_path, rank=0, load_epoch=1, embedding_weight_name='embed_weight'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.embedding_save_path = embedding_save_path
        self.rank = rank
        self.load_epoch = load_epoch
        self.embedding_weight_name = embedding_weight_name

    @property
    @memoized
    def _vocab(self):
        return load_vocab(self.vocab_path)

    @property
    @memoized
    def _embedding(self):
        # load model
        _, arg_params, _ = load_model(self.model_path, self.rank, self.load_epoch)
        assert self.embedding_weight_name in arg_params, "{} parameter not in the w2v model, " \
                                                         "please check the embedding weight name".format(
                                                          self.embedding_weight_name)

        return arg_params.get(self.embedding_weight_name)

    def dumper(self):

        vocab = self._vocab
        assert vocab is not None and len(vocab) > 0, "vocabulary can not be empty"
        embed = self. _embedding
        assert embed is not None, "Failed to load the embedding"
        embed_np = embed.asnumpy()
        word_num, embed_size = embed_np.shape
        w2v = dict()
        for key, value in vocab.iteritems():
            w2v.setdefault(key, embed_np[value])



if __name__ == '__main__':
    W2vDumper('../data/model/query2vec', '../data/word2vec/vocab.pkl', '../data/word2vec/embed.pkl').dumper()

