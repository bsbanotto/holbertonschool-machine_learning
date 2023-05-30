#!/usr/bin/env python3
"""
Perform k-means on a dataset. Only import allowed is sklearn.cluster
"""
import sklearn.cluster


def kmeans(X, k):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: number of clusters
    Returns: C, clss
        C: numpy.ndarray shape (k, d) containing the centroid means
        clss: numpy.ndarray shape (n,) containing the index of the cluster
            C that each data point belongs to
    """
    C, clss, _ = sklearn.cluster.k_means(X, k, n_init='auto')

    return C, clss
