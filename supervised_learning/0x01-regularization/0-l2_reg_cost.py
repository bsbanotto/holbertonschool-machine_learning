#!/usr/bin/env python3
"""
Function that calculates the cost of a neural network with L2 Regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the neural network
    m: number of data points used
    Returns the cost of the network accounting for L2 regularization
    """
    Frobenius_Norm = []
    for key, value in weights.items():
        if 'W' in key:
            Frobenius_Norm.append(np.sum(value * value))
    L2_norm = np.sum(Frobenius_Norm)
    return cost + (lambtha / (2 * m)) * L2_norm
