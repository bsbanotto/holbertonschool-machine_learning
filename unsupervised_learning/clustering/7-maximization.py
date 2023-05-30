#!/usr/bin/env python3
"""
Calculate the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    X: numpy.ndarray shape (n, d) containing the data set
    g: numpy.ndarray shape (k, n) containing the posterior probabilities for
        each datapoint in each cluster
    Returns pi, m, S or None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing the updated priors for each
            cluster
        m: numpy.ndarray of shape (k, d) containing the updated centroid means
            for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the updated covariance
            matrices for each cluster
    """
    # If guard agains bad data input
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if np.sum(g.shape[0] != 0):
        return None, None, None
    return("hello")
