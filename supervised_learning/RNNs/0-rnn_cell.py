#!/usr/bin/env python3
"""
Module to implement a simple RNN
"""
import numpy as np


class RNNCell():
    """
    Class that represents a simple RNN
    """
    def __init__(self, i, h, o):
        """
        RNCell Class Constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs

        Creates the public instance attributes Wh, Wy, bh, by that represent
            the weights and biases of the cell
            Wh and bh: concatenated cell hidden state and cell input data
            Wy and by: for the cell output

        Weights should be initialized using a random distribution in the order
            listed
        Weights will be used on the right side for matrix multiplication
        Biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step
        Args:
            h_prev: np.ndarray shape(m, h) containing the previous hidden state
            x_t: np.ndarray shape(m, i) contains the data input for the cell
                m: batch size for the data

        Returns:
            h_next: the next hidden state
            y: the output of the cell which should use a softmax activation
        """
        # Previous hidden layer and input data are what we put in
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state is tanh of (cell_input * weights + bias)
        h_next = np.tanh(np.matmul(cell_input, self.Wh) + self.bh)

        # Compute the cell output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
