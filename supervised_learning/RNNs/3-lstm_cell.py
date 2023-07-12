#!/usr/bin/env python3
"""
Represents an LSTM unit
"""
import numpy as np


class LSTMCell():
    """
    Class that represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Creates public instance attributes Wf Wu Wc Wo Wy bf bu bc bo by that
            represent the weights and biases of the cell
            Wf, bf - forget gate
            Wu, bu - update gate
            Wc, bc - intermediate cell state
            Wo, bo - output gate
            Wy, by - outputs

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Helper function for clean code in forward
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: np.ndarray shape(m, h) containing the previous hidden state
            c_prev: np.ndarray shape(m, h) containing the previous cell state
            x_t: np.ndarray shape (m, i) that contains the input for the cell
                m: match size for the data

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        # Previous hidden layer and input data are what we put in
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget Gate
        f = self.sigmoid(np.dot(cell_input, self.Wf) + self.bf)
        # Update Gate
        u = self.sigmoid(np.dot(cell_input, self.Wu) + self.bu)
        # Output Gate
        o = self.sigmoid(np.dot(cell_input, self.Wo) + self.bo)

        # Intermediate Cell State
        c_intermediate = np.tanh(np.dot(cell_input, self.Wc) + self.bc)

        # Next Cell
        c_next = c_prev * f + u * c_intermediate

        # Next Hidden State
        h_next = o * np.tanh(c_next)

        # Compute Output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
