# -*- coding: utf-8 -*-
from network.seq2seq.encoder import BiDirectionalLstmEncoder
from network.seq2seq.decoder import LstmDecoder

import mxnet as mx
from collections import namedtuple
from itertools import chain
from utils.decorator_util import memoized
from model import Model

encoder_parameter = namedtuple('encoder_parameter', 'encoder_layer_num encoder_vocab_size '
                                                    'encoder_hidden_unit_num encoder_embed_size encoder_dropout')

decoder_parameter = namedtuple('decoder_parameter', 'decoder_layer_num decoder_vocab_size '
                                                    'decoder_hidden_unit_num decoder_embed_size decoder_dropout')

data_label_names_parameter = namedtuple('data_label_names_parameter', 'data_names label_names')


class Seq2seqModel(Model):
    def __init__(self, encoder_para, decoder_para, data_label_names_para, share_embed=True):
        self.encoder_para = encoder_para
        self.decoder_para = decoder_para
        self.data_label_names_para = data_label_names_para
        self.share_embed = share_embed
        self._initialize()

    def _initialize(self):
        assert isinstance(self.encoder_para, encoder_parameter)
        assert isinstance(self.decoder_para, decoder_parameter)

        for (parameter, value) in chain(self.encoder_para._asdict().iteritems(),
                                        self.decoder_para._asdict().iteritems(),
                                        self.data_label_names_para._asdict().iteritems()):
            setattr(self, parameter, value)

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

    def unroll(self, encoder_seq_len, decoder_seq_len):
        encoder = self.encoder(encoder_seq_len)
        decoder = self.decoder(decoder_seq_len)

        forward_hidden_all, backward_hidden_all, _, _ = encoder.encode()

        decoded_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='decoded_init_state')
        decoder_softmax = decoder.decode(decoded_init_state)
        return decoder_softmax

    def network_symbol(self):
        def _sym_gen(bucket_key):
            sym = self.unroll(bucket_key[0], bucket_key[1])
            return sym, self.data_names, self.label_names

        return _sym_gen
