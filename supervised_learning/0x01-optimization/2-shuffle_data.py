#!/usr/bin/env python3
"""
Module that contains a function that shuffles the data in two matrices
in the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    X: First numpy.ndarray matrix of shape (m, nx) to be shuffled
    Y: Second numpy.ndarray matrix of shape (m, ny) to be shuffled
    m: number of data points
    nx/ny: number of features in X and Y respectively
    """
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]
