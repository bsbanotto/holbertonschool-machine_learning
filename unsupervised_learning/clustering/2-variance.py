#!/usr/bin/env python3
"""
Find intra-cluster variance on a data set
"""
import numpy as np


def variance(X, C):
    """
    Function that calculates the total intra-cluster variance
    X: numpy.ndarray shape (n, d) containing the dataset
    C: numpy.ndarray shape (k, d) containing centroid means for each cluster
    Returns var, or None on failure
        var is the total variance
    """
    # Guard against bad input data
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None

    if X.shape != C.shape:
        return None
    # Calculate distance between points and cluster centroids
    min_dist = np.min((np.linalg.norm((X - C[:, None, :]), axis=2).T), axis=1)
    var = np.sum(np.square(min_dist))

    return var
