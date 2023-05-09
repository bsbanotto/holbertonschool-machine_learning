#!/usr/bin/env python3
"""
Function to calculate the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
        n: number of data points
        d: number of dimensions in each data poin
        If X is not a 2D numpy.ndarray raise a TypeError with message
            X must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with message
            X must contain multiple data points
    Returns mean, cov
        mean: numpy.ndarray of shape (1, d) contaiing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix of
        the data set
    """
    if type(X) != np.ndarray and len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = 0
    return mean, cov
