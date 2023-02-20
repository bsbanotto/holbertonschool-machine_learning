#!/usr/bin/env python3
"""
Function that normalizes an unactivated output of a neural network using
batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a neural network using
    batch normalization
    Z: numpy.ndarray of shame (m, n) that is to be normalized
        m: number of data points
        n: number of features in Z
    gamma: numpy.ndarray of shape (1, n) containing the scales used for batch
        normalization
    beta: numpy.ndarray of shape (1, n) containing the offsets used for batch
        normalization
    epsilon: small number used to avoid divide by zero errors
    Returns a normalized Z matrix
    """
    mu = np.mean(Z, axis=0)
    sigma = np.std(Z, axis=0)
    Z_norm = (Z - mu) / ((sigma ** 2 + epsilon) ** (1/2))
    return (gamma * Z_norm) + beta
