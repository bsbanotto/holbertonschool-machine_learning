#!/usr/bin/env python3
"""
Function to perform forward propagation for a bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation over t time steps of a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell
        X: np.ndarray of shape (t, m, i) that is the data to be used
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state in the forward direction, given as a
            np.ndarray of shape (m, h)
            h: dimensionality of the hidden state
        h_t: initial hidden state in the backward direction, given as a
            np.ndarray of shape (m, h)
            h: dimensionality of the hidden state

    Returns:
        H: np.ndarray containing all of the concatenated hidden states
        Y: np.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H_forward = np.zeros((t + 1, m, h))
    H_forward[0] = h_0

    H_backward = np.zeros((t + 1, m, h))
    H_backward[0] = h_t

    # Loop through all time steps building forward and backward hidden states
    for time in range(t):
        # Do the forward bit
        H_forward[time + 1] = bi_cell.forward(H_forward[time], X[time])

        # Do the backward bit
        H_backward[time + 1] = bi_cell.backward(H_backward[time], X[time])

    # Concatenate H_forward and H_backward
    H = np.concatenate((H_forward[1:], H_backward[:-1]), axis=2)

    # Calculate the outputs of the network
    Y = bi_cell.output(H)

    return H, Y
