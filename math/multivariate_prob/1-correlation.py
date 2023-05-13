#!/usr/bin/env python3
"""
Function to calculate a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    C: numpy.ndarray shape (d, d) containing a covariance matrix
        d: number of dimensions
        If C is not a numpy.ndarray, raise a TypeError with message
            C must be a numpy.ndarray
        If C does not have shape (d, d) raise a ValueError with message
            C must be a 2D square matrix
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) < 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    vars = np.diag(np.diag(C))
    std_dev = np.sqrt(vars)
    inv = np.linalg.inv(std_dev)

    corr_matrix = np.matmul(np.matmul(inv, C), inv)

    return corr_matrix
