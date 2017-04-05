import mxnet as mx
from collections import namedtuple


LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])


def lstm(num_hidden, indata, prev_state, param, seqid, layerid, dropout=0.):
    """Long-Short Term Memory (LSTM) network cell
    Parameters
    ----------
        num_hidden : int
            number of units in output symbol
        indata : sym.Variable
            input symbol, 2D, batch * num_hidden
        prev_state : sym.Variable
            state from previous step
        param: LSTMParam
            namedtuple for weight sharing between cells.
        seqid: int
            sequence id
        layerid:
            layer id
        dropout: float
            the probability to ignore the neuron outputs

    Returns
    -------
        LSTMState
    """
    # TODO remove this cell, will use the mx build-in cell in next version
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqid, layerid))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqid, layerid))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqid, layerid))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)
