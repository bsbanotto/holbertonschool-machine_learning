#!/usr/bin/env python3
"""
Function that performs the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
        for the algorithm
    tol: non-negative float containing the tolerance of the log likelihood,
        used to determine early stopping. If the difference is less than or
        equal to tol, stop the algorithm
    verbose: boolean that determines if information should be printed about the
        algorithm
        If true: print `log Likelihood after {i} iterations: {l} after every 10
            iterations and after the last
            {i} number of iterations of the EM algorithm
            {l} log likelihood, rounded to 5 decimal places
    Returns: pi, m, S, g, l or None, None, None, None, None on failure
        pi: numpy.ndarray shape (k,) containing the priors for each cluster
        m: numpy.ndarray shape (k, d) containing the centroid means for each
            cluster
        S: numpy.ndarray shape (k, d, d) containing the covariance matrices for
            each cluster
        g: numpy.ndarray shape (k, n) containing the probabilities for each
            data point in each cluster
        l: the log likelihood of the model
    """
    # Guard against bad arguments
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    l_start = 0

    for i in range(iterations):
        g, l_finish = expectation(X, pi, m, S)

        # Check to see if l_start and l_finish are within tol
        if abs(l_finish - l_start) <= tol:
            if verbose:
                l_round = round(l_finish, 5)
                print("Log Likelihood after {} iterations: {}".format(i,
                                                                      l_round))
            return pi, m, S, g, l_start

        pi, m, S = maximization(X, g)

        if verbose and i % 10 == 0:
            l_round = round(l_finish, 5)
            print("Log Likelihood after {} iterations: {}".format(i, l_round))

        l_start = l_finish

    return pi, m, S, g, l_finish
