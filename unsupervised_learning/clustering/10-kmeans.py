#!/usr/bin/env python3
"""
Perform k-means on a dataset. Only import allowed is sklearn.cluster
"""
from sklearn.cluster import k_means


def kmeans(X, k):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: number of clusters
    Returns: C, clss
        C: numpy.ndarray shape (k, d) containing the centroid means
        clss: numpy.ndarray shape (n,) containing the index of the cluster
            C that each data point belongs to
    """
    C, clss, _ = k_means(X, k)

    return C, clss
