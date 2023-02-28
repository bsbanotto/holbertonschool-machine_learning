#!/usr/bin/env python3
"""
Function that conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X: numpy.ndarray of shape (nx, m) containing the input data for the network
        nx: number of input features
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the network
    keep_prob: the probability that a node will be kept
    All layers except the last should use the tanh activation function
    Last layer will use the softmax activation function
    Returns a dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    cache = {}
    A = X
    cache["A{}".format(0)] = X
    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, A) + b
        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
            cache["A{}".format(layer)] = A
        else:
            A = np.tanh(Z)
            dropout_mask = np.random.binomial(n=1, p=keep_prob, size=A.shape)
            A = A * dropout_mask / keep_prob
            cache["A{}".format(layer)] = A
            cache["D{}".format(layer)] = dropout_mask
    return cache
