#!/usr/bin/env python3
"""
Function that updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m) that contains data labels
        classes: number of classes
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs and dropout masks of each layer of the nn
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network
    All layers except the last use the tanh activation function
    The last layer uses the softmax activation function
    The weights of the network should be updated in place
    """
    m = Y.shape[1]

    for layer in range(L, 0, -1):
        A = cache["A{}".format(layer)]
        A_prev = cache["A{}".format(layer - 1)]

        if layer != L:
            dZ = dA_prev * (1 - (A ** 2))
            dZ = (dZ * cache["D{}".format(layer)]) / keep_prob
        else:
            dZ = A - Y
        W = weights["W{}".format(layer)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.matmul(W.T, dZ)

        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db
