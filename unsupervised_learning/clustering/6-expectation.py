#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X: numpy.ndarray of shape (n, d) containing the data set
    pi: numpy.ndarray of shape (k,) containing the priors for each cluster
    m: numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster
    S: numpy.ndarray of shape (k, d, d) containing the covariance matrices for
        each cluster
    Returns g, l or None, None on failure
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
            for each data point in each cluster
        loglikelihood: total log likelihood
    """
    # Guard against bad data input
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or m.shape[1] != S.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or m.shape[0] != S.shape[0]:
        return None, None
    if np.sum(pi) != 1:
        return None, None

    n = X.shape[0]
    k = m.shape[0]

    g = np.zeros((k, n))

    # Calculate PDF of each data point in each cluster
    PDF = np.zeros((k, n))
    for i in range(k):
        PDF[i] = pi[i] * pdf(X, m[i], S[i])

    # Normalize g
    g = PDF / np.sum(PDF, axis=0)

    loglikelihood = np.sum(np.log(np.sum(PDF, axis=0)))

    return g, loglikelihood
