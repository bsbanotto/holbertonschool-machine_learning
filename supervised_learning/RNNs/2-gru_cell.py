#!/usr/bin/env python3
"""
Gated Recurrent Unit
"""
import numpy as np


class GRUCell():
    """
    Class that represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Creates public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, bz that
            represent the weights and biases of the cell
            Wz, bz - update gate
            Wr, br - reset gate
            Wh, bh - intermediate hidden state
            Wy, by - output

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Helper function for clean code in forward
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: np.ndarray shape(m, h) containing the previous hidden state
            x_t: np.ndarray shape(m, i) containing the data input for the cell
                m: batch size for the data

        Returns:
            h_next: the next hidden state
            y: the output of the cell
        """
        # Previous hidden layer and input data are what we put in
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate
        z = self.sigmoid(np.dot(cell_input, self.Wz) + self.bz)
        # Reset Gate
        r = self.sigmoid(np.dot(cell_input, self.Wr) + self.br)

        # Hidden State
        reset_input = np.concatenate((r * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(np.dot(reset_input, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_intermediate

        # Get the output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
