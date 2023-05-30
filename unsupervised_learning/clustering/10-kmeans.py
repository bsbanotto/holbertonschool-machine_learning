#!/usr/bin/env python3
"""
Perform k-means on a dataset. Only import allowed is sklearn.cluster
"""
import sklearn.cluster as Cluster


def kmeans(X, k):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: number of clusters
    Returns: C, clss
        C: numpy.ndarray shape (k, d) containing the centroid means
        clss: numpy.ndarray shape (n,) containing the index of the cluster
            C that each data point belongs to
    """
    model = Cluster.KMeans(n_clusters=k, n_init='auto').fit(X)

    return model.cluster_centers_, model.labels_
