# coding=utf-8
# pylint: disable=import-error, too-many-arguments, too-many-instance-attributes
"""Io iter for seq2seq model"""
import mxnet as mx


class DataBatch(object):
    """Object for holding a mini-batch of data and related information."""

    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0

    @property
    def provide_data(self):
        """The name and shape of data of this batch"""
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        """The name and shape of label of this batch"""
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class Seq2seqMaskedBucketIoIter(mx.io.DataIter):
    """DataIter to given number of batches per epoch for seq2seq model.
       Every batch contains encoder, decoder data, mask data and bucket key.

       Parameters
       ----------
            data_stream: object with iterator
                 data stream to generator which generate data(encoder_data, encoder_mask_data,
                 decoder_data, decoder_mask_data, label, bucket)
            encoder_init_states: list
                 init state for encoder. e.g. ['forward_encoder_l0_init_c',
                                                (batch_size, encoder_hidden_unit_num)]
            decoder_init_states: list
                 init state for decoder. e.g. ['decoder_l0_init_c',
                                                (batch_size, decoder_hidden_unit_num)]
            default_bucket_key: tuple
                 The key for the default bucket. Usually, it should be the max bucket
            batch_size: int
                 batch size for each databatch
            encoder_data_name: str
                data name for encoder which is in accordance with the data name in encoder
            decoder_data_name: str
                data name for decoder which is in accordance with the data name in decoder
            encoder_mask_name: str
                masked data name for encoder
            decoder_mask_name: str
                masked data name for decoder
            label_name : str
                name of label
        Notes
        -----
        More detail: http://mxnet.io/tutorials/python/data.html
    """

    def __init__(self, data_stream, encoder_init_states, decoder_init_states,
                 default_bucket_key, batch_size,
                 encoder_data_name='encoder_data', encoder_mask_name='encoder_mask',
                 decoder_data_name='decoder_data', decoder_mask_name='decoder_mask',
                 label_name='decoder_softmax_label'):
        self.data_stream = data_stream

        self.encoder_data_name = encoder_data_name
        self.decoder_data_name = decoder_data_name
        self.label_name = label_name

        self.encoder_mask_name = encoder_mask_name
        self.decoder_mask_name = decoder_mask_name
        self.encoder_init_states = encoder_init_states
        self.decoder_init_states = decoder_init_states
        self.encoder_init_state_arrays = [mx.nd.zeros(x[1]) for x in encoder_init_states]
        self.decoder_init_state_arrays = [mx.nd.zeros(x[1]) for x in decoder_init_states]

        self.encoder_init_state_names = [x[0] for x in encoder_init_states]
        self.decoder_init_state_names = [x[0] for x in decoder_init_states]

        self.default_bucket_key = default_bucket_key
        self.provide_data = [(encoder_data_name, (batch_size, self.default_bucket_key[0])),
                             (encoder_mask_name, (batch_size, self.default_bucket_key[0])),
                             (decoder_data_name, (batch_size, self.default_bucket_key[1])),
                             (decoder_mask_name, (batch_size, self.default_bucket_key[1]))] \
                            + encoder_init_states + decoder_init_states
        self.provide_label = [(label_name, (batch_size, self.default_bucket_key[1]))]

    @property
    def data_names(self):
        """The name and shape of data provided by this iterator"""
        # pylint: disable=line-too-long
        return [self.encoder_data_name] + [self.encoder_mask_name] + [self.decoder_data_name] + \
               [self.decoder_mask_name] + self.encoder_init_state_names + self.decoder_init_state_names

    @property
    def label_names(self):
        """The name and shape of label provided by this iterator"""
        return [self.label_name]

    def __iter__(self):
        # pylint: disable=line-too-long
        for encoder_data, encoder_mask_data, decoder_data, decoder_mask_data, label, bucket in self.data_stream:
            data_all = [mx.nd.array(encoder_data), mx.nd.array(encoder_mask_data)] + \
                       [mx.nd.array(decoder_data), mx.nd.array(decoder_mask_data)] + \
                       self.encoder_init_state_arrays + self.decoder_init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = [self.encoder_data_name, self.encoder_mask_name] + [
                self.decoder_data_name, self.decoder_mask_name] + \
                         self.encoder_init_state_names + self.decoder_init_state_names
            label_names = [self.label_name]
            data_batch = DataBatch(data_names, data_all, label_names, label_all,
                                   bucket)
            yield data_batch

    def reset(self):
        """Reset the data stream iterator. """
        self.data_stream.reset()
