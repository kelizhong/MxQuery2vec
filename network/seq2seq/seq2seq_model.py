# -*- coding: utf-8 -*-
from network.seq2seq.encoder import BiDirectionalLstmEncoder
from network.seq2seq.decoder import LstmDecoder

import mxnet as mx
from collections import namedtuple
from itertools import chain
from utils.decorator_util import memoized
from model import Model
'''
Papers:
[1] Neural Machine Translation by Jointly Learning to Align and Translate(https://arxiv.org/pdf/1409.0473v6.pdf)
'''


encoder_parameter = namedtuple('encoder_parameter', 'encoder_layer_num encoder_vocab_size '
                                                    'encoder_hidden_unit_num encoder_embed_size encoder_dropout')

decoder_parameter = namedtuple('decoder_parameter', 'decoder_layer_num decoder_vocab_size '
                                                    'decoder_hidden_unit_num decoder_embed_size decoder_dropout')


class Seq2seqModel(Model):
    """Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    Parameters
    ----------
    encoder_para: namedtuple
        encoder parameter
    decoder_para: namedtuple
        decoder parameter
    share_embed: bool
        whether share embedding weight
    """

    def __init__(self, encoder_para, decoder_para, share_embed=True):
        self.encoder_para = encoder_para
        self.decoder_para = decoder_para
        self.share_embed = share_embed
        self._initialize()

    def _initialize(self):
        assert isinstance(self.encoder_para, encoder_parameter)
        assert isinstance(self.decoder_para, decoder_parameter)

        for (parameter, value) in chain(self.encoder_para._asdict().iteritems(),
                                        self.decoder_para._asdict().iteritems()):
            setattr(self, parameter, value)

    @memoized
    def get_init_state_shape(self, batch_size):
        """initalize states for LSTM"""

        encoder_init_states = BiDirectionalLstmEncoder.get_init_state_shape(batch_size, self.encoder_layer_num,
                                                                            self.encoder_hidden_unit_num)

        decoder_init_states = LstmDecoder.get_init_state_shape(batch_size, self.decoder_layer_num,
                                                               self.decoder_hidden_unit_num)
        return encoder_init_states, decoder_init_states

    @property
    @memoized
    def embed_weight(self):
        # embedding
        if self.share_embed:
            embed_weight = mx.sym.Variable("share_embed_weight")
        else:
            embed_weight = None
        return embed_weight

    def encoder(self, seq_len):
        encoder = BiDirectionalLstmEncoder(seq_len=seq_len, use_masking=True,
                                           hidden_unit_num=self.encoder_hidden_unit_num,
                                           vocab_size=self.encoder_vocab_size, embed_size=self.encoder_embed_size,
                                           dropout=self.encoder_dropout, layer_num=self.encoder_layer_num,
                                           embed_weight=self.embed_weight)
        return encoder

    def decoder(self, seq_len):
        decoder = LstmDecoder(seq_len=seq_len, use_masking=True, hidden_unit_num=self.decoder_hidden_unit_num,
                              vocab_size=self.decoder_vocab_size, embed_size=self.decoder_embed_size,
                              dropout=self.decoder_dropout,
                              layer_num=self.decoder_layer_num, embed_weight=self.embed_weight)
        return decoder

    def attention(self, encoder_seq_len):
        """ base on [1] """
        # TODO add attention mechanism
        return None

    def unroll(self, encoder_seq_len, decoder_seq_len):
        encoder = self.encoder(encoder_seq_len)
        decoder = self.decoder(decoder_seq_len)

        encoder_last_state = encoder.encode()
        decoder_softmax = decoder.decode(encoder_last_state)
        return decoder_softmax

    def network_symbol(self, data_names, label_names):
        def _sym_gen(bucket_key):
            sym = self.unroll(bucket_key[0], bucket_key[1])
            return sym, data_names, label_names

        return _sym_gen
