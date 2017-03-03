from network.seq2seq.encoder import BiDirectionalLstmEncoder
from network.seq2seq.decoder import LstmDecoder

import mxnet as mx


def s2s_unroll(source_layer_num, source_seq_len, source_vocab_size, source_hidden_unit_num, source_embed_size,
               source_dropout,
               target_layer_num, target_seq_len, target_vocab_size, target_hidden_unit_num, target_embed_size,
               target_dropout):
    # embedding

    embed_weight = mx.sym.Variable("embed_weight")

    encoder = BiDirectionalLstmEncoder(seq_len=source_seq_len, use_masking=True, hidden_unit_num=source_hidden_unit_num,
                                       vocab_size=source_vocab_size, embed_size=source_embed_size,
                                       dropout=source_dropout, layer_num=source_layer_num, embed_weight=embed_weight)

    decoder = LstmDecoder(seq_len=target_seq_len, use_masking=True, hidden_unit_num=target_hidden_unit_num,
                          vocab_size=target_vocab_size, embed_size=target_embed_size, dropout=target_dropout,
                          layer_num=target_layer_num, embed_weight=embed_weight)
    forward_hidden_all, backward_hidden_all, source_representations, source_mask_sliced = encoder.encode()

    encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='encoded_for_init_state')
    target_representation = decoder.decode(encoded_for_init_state)
    return target_representation


def sym_gen(source_vocab_size=None, source_layer_num=None, source_hidden_unit_num=None, source_embed_size=None,
            source_dropout=None,
            target_vocab_size=None, target_layer_num=None, target_hidden_unit_num=None, target_embed_size=None,
            target_dropout=None,
            data_names=None, label_names=None):
    def _sym_gen(bucket_key):
        sym = s2s_unroll(source_layer_num=source_layer_num, source_seq_len=bucket_key[0],
                         source_vocab_size=source_vocab_size,
                         source_hidden_unit_num=source_hidden_unit_num, source_embed_size=source_embed_size,
                         source_dropout=source_dropout,
                         target_layer_num=target_layer_num, target_seq_len=bucket_key[1],
                         target_vocab_size=target_vocab_size,
                         target_hidden_unit_num=target_hidden_unit_num, target_embed_size=target_embed_size,
                         target_dropout=target_dropout)
        return sym, data_names, label_names

    return _sym_gen
