import mxnet as mx


class Word2vec(object):
    """The model of word2vec, using nce loss
    Parameters
    ----------
    batch_size: int
        batch size for each data batch
    vocab_size: int
        the size of vocabulary of the corpus
    embed_size: int
        word embedding size
    window_size: int
        the maximum distance between the current and predicted word within a sentence
    """

    def __init__(self, batch_size, vocab_size, embed_size, window_size):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.vocab_size = vocab_size

    def network_symbol(self):
        """word2vec network symbol for training"""
        input_num = 2 * self.window_size
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')
        label_weight = mx.sym.Variable('label_weight')
        embed_weight = mx.sym.Variable('embed_weight')
        data_embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                      weight=embed_weight,
                                      output_dim=self.embed_size, name='data_embed')
        datavec = mx.sym.SliceChannel(data=data_embed,
                                      num_outputs=input_num,
                                      squeeze_axis=1, name='data_slice')
        pred = datavec[0]
        for i in range(1, input_num):
            pred = pred + datavec[i]

        return self.nce_loss(data=pred,
                             label=label,
                             label_weight=label_weight,
                             embed_weight=embed_weight)

    def nce_loss(self, data, label, label_weight, embed_weight):
        """Noise contrastive Estimation"""
        label_embed = mx.sym.Embedding(data=label, input_dim=self.vocab_size,
                                       weight=embed_weight,
                                       output_dim=self.embed_size, name='label_embed')
        data = mx.sym.Reshape(data=data, shape=(-1, 1, self.embed_size))
        pred = mx.sym.broadcast_mul(data, label_embed)
        pred = mx.sym.sum(data=pred, axis=2)
        return mx.sym.LogisticRegressionOutput(data=pred,
                                               label=label_weight)

    def embedding_weight_symbol(self):
        data = mx.sym.Variable('data')
        embed_weight = mx.sym.Variable('embed_weight')
        data_embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                      weight=embed_weight,
                                      output_dim=self.embed_size, name='data_embed')
        return data_embed
