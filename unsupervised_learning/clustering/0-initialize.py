#!/usr/bin/env python3
"""
Initialze K-means algorithm
"""
import numpy as np


def initialize(X, k):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset used for clustering
        n: number of data points
        d: number of dimensions for each data point
    k: positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate
    uniform distribution along each dimension in d
        The minimum and maximum values for the distribution should be the
        minimum and maximum values of X along each dimension in d
    Should use numpy.random.uniform exactly once
    Returns a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    try:
        if k <= 0:
            return None

        min_val = X.min(axis=0)
        max_val = X.max(axis=0)

        centroids = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))

        return centroids
    except Exception:
        return None
