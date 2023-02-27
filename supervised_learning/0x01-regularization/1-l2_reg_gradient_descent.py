#!/usr/bin/env python3
"""
Function that updates the weights and biases of a neural network using
gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m) that contains the correct
    labels for the data
        classes: number of classes
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs of each layer of the neural network
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers of the network
    The neural network uses tanh activations on each layer except the last
    The last layer uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]

    for layer in range(L, 0, -1):
        A_layer = cache["A{}".format(layer)]
        A_layer_prev = cache["A{}".format(layer - 1)]

        if layer == L:
            dz_layer = (A_layer - Y)
        else:
            dz_layer = dA_layer_prev * (1 - np.square(A_layer))

        W_layer = weights["W{}".format(layer)]

        dA_layer_prev = np.matmul(W_layer.T, dz_layer)

        dW_layer = (np.matmul(dz_layer, A_layer_prev.T) / m) +\
                   ((lambtha / m) * W_layer)
        db_layer = np.sum(dz_layer, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] -= alpha * dW_layer
        weights["b{}".format(layer)] -= alpha * db_layer
