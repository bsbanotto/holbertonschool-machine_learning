#!/usr/bin/env python3
"""
Perform forward prop for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward prop for a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l that will be used for
            the forward propagation
            l: number of layers
        X: data to be used, given as np.ndarray shape (t, m, i)
            t: max number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: the initial hidden state given as np.ndarray shape (l, m, h)
            l: dimensionality of the hidden state

    Returns:
        H: np.ndarray containing all of the hidden states
        Y: np.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    o = rnn_cells[-1].Wy.shape[1]
    Y = np.zeros((t, m, o))

    for time in range(t):
        x_t = X[time]
        for layer in range(layers):
            rnn_cell = rnn_cells[layer]
            h_prev = H[time, layer]
            h_next, Y[time] = rnn_cell.forward(h_prev, x_t)
            H[time + 1, layer] = h_next
            x_t = h_next

    return H, Y
