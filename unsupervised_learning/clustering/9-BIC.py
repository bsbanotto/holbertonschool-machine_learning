#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using the Bayesian Information
Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer containing minimum number of clusters to check for
    kmax: positive integer containing maximium number of clusters to check for
        if kmax is none, it should be set to the max number possible
    iterations: positive integer containing the max number of iterations
    tol: non-negative float containing the tolerance for the EM algorithm
    verbose: boolesn that determines if the EM algorithm should print
        information to stdout
    Returns: best_k, best_result, l, b or None, None, None, None on failure
        best_k: best value for K based on its BIC
        best_result: tuple containing pi, m, S
        l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested
            use BIC = p * ln(n) - 2 * l
            p: number of parameters required for the model
            n: number of data points used to create the model
            l: log likelihood of the model
    """
    # Guard against bad argument inputs
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
        if kmax <= 0 or kmax <= kmin:
            return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    return ("hello")
