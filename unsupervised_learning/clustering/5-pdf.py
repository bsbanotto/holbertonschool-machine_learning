#!/usr/bin/env python3
"""
Calculate the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    X: numpy.ndarray shape (n, d) containing the data points whose PDF should
        be evaluated
    m: numpy.ndarray shape (d,) containing the mean of the distribution
    S: numpy.ndarray shape (d, d) containing the covariance of the distribution
    Returns: P or None on failure
        P: numpy.ndarray of shape (n,) containing the PDF values for each
            data point
        All values in P should have a minimum value of 1e-300
    """
    # If guard against incorrect arguments
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    n, _ = X.shape

    P = np.zeros((n))

    return P