#!/usr/bin/env python3
"""
Method to perform forward prop for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Function to perform forward propagation for an RNN
    Args:
        rnn_cell: an instance of RNNCell that will be used for forward prop
        X: data to be used, np,ndarray shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: the initial hidden state, np.ndarray shape (m, h)

    Returns:
        H: np.ndarray containing all of the hidden states
        Y: np.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    o = rnn_cell.Wy.shape[1]
    Y = np.zeros((t, m, o))

    for time in range(t):
        H[time + 1], Y[time] = rnn_cell.forward(H[time], X[time])

    return H, Y
