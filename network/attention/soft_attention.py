import mxnet as mx


class SoftAttention:
    """This is an attention Seq2seq model based on
    [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473v6.pdf)
    The math:
        Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X);
            so H is a sequence of vectors of length m.
    """
    def __init__(self, seq_len, encoder_hidden_output_dim, state_dim):
        self.e_weight_W = mx.sym.Variable('energy_W_weight', shape=(state_dim, state_dim))
        self.e_weight_U = mx.sym.Variable('energy_U_weight', shape=(encoder_hidden_output_dim, state_dim))
        self.e_weight_v = mx.sym.Variable('energy_v_bias', shape=(state_dim, 1))
        self.seq_len = seq_len
        self.encoder_hidden_output_dim = encoder_hidden_output_dim
        self.state_dim = state_dim

    def attend(self, encoder_hidden_all, concat_attended, state, attend_masks, use_masking):
        '''
        Encoder:
            X = Input Sequence of length m.
            encoder_hidden_all = Bidirection_LSTM(X);
            so encoder_hidden_all is a sequence of hidden vectors of length m.
        v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
        The weight alpha[i, j] for each hj is computed as follows:
        energy = a(s(i-1), H(j))
        alhpa = softmax(energy)
        Where a is a feed forward network.
        Parameters
        ----------
            encoder_hidden_all: list [seq_len, (batch, encoder_hidden_output_dim)]
            concat_attended:  (batch, seq_len, encoder_hidden_output_dim )
            state: (batch, state_dim)
            attend_masks: list [seq_len, (batch, 1)]
            use_masking: boolean
        Returns
        -------
        '''
        energy_all = []
        pre_compute = mx.sym.dot(state, self.e_weight_W, name='_energy_0')
        for idx in range(self.seq_len):
            h = encoder_hidden_all[idx]  # (batch, attend_dim)
            energy = pre_compute + mx.sym.dot(h, self.e_weight_U,
                                              name='_energy_1_{0:03d}'.format(idx))  # (batch, state_dim)
            energy = mx.sym.Activation(energy, act_type="tanh",
                                       name='_energy_2_{0:03d}'.format(idx))  # (batch, state_dim)
            energy = mx.sym.dot(energy, self.e_weight_v, name='_energy_3_{0:03d}'.format(idx))  # (batch, 1)
            if use_masking:
                energy = energy * attend_masks[idx] + (1.0 - attend_masks[idx]) * (-10000.0)  # (batch, 1)
            energy_all.append(energy)

        all_energy = mx.sym.Concat(*energy_all, dim=1, name='_all_energy_1')  # (batch, seq_len)

        alpha = mx.sym.SoftmaxActivation(all_energy, name='_alpha_1')  # (batch, seq_len)
        alpha = mx.sym.Reshape(data=alpha, shape=(-1, self.seq_len, 1),
                               name='_alpha_2')  # (batch, seq_len, 1)

        weighted_attended = mx.sym.broadcast_mul(alpha, concat_attended,
                                                 name='_weighted_attended_1')  # (batch, seq_len, attend_dim)
        weighted_attended = mx.sym.sum(data=weighted_attended, axis=1,
                                       name='_weighted_attended_2')  # (batch,  attend_dim)
        return weighted_attended
