#!/usr/bin/env python3
"""
Function to test for the optimum number of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X: numpy.ndarray shape (n, d) containing the data set
    kmin: positive integer containing minimum number of clusters to check for
    kmax: positive integer containing maximum number of clusters to check for
    iterations: positive integer containing the max number of iterations
    Returns results, d_vars or None, None
        results: list containing outputs of the K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
            cluster size for each cluster size
    """
    # If guard against bad input data
    if type(kmin) or type(kmax) is not int:
        return None, None
    
    if kmin <= 0 or kmax <= 0:
        return None, None

    if kmax <= kmin:
        return None, None

    if type(X) is not np.ndarray:
        return None, None

    if type(iterations) is not int:
        return None, None

    if iterations <= 0:
        return None, None

    # Create empty lists
    results = []
    vars = []
    d_vars = []

    # Calculate kmeans and variance through range of kmin to kmax
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vars.append(variance(X, C))

    """Calculate d_vars from the smallest cluster size for each cluster"""
    for x in vars:
        d_vars.append(vars[0] - x)

    return results, d_vars
