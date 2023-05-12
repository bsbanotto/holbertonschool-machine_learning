#!/usr/bin/env python3
"""
Function to calculate the mean and covariance of a data set
mean is average
covariance is the degree to which random variables behave in a similar way
"""
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
        n: number of data points
        d: number of dimensions in each data point
        If X is not a 2D numpy.ndarray raise a TypeError with message
            X must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with message
            X must contain multiple data points
    Returns mean, cov
        mean: numpy.ndarray of shape (1, d) contaiing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix of
        the data set
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape

    mean = np.mean(X, axis=0, keepdims=True)
    # This section deviated from required answers after the third decimal
    # Will refactor to get what the checker wants, but I like this solution
    """
        cov = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                cov[i][j] = np.mean(X[:, i] * X[:, j]) - mean[0][i] * mean[0][j]
        return mean, cov
    """
    dev = X - mean
    cov = np.matmul(dev.T, dev)
    cov = cov / (X.shape[0] - 1)
    return mean, cov
