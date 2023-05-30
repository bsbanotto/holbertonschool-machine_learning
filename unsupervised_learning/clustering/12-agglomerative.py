#!/usr/bin/env python3
"""
Perform agglomerative clustering with Ward linkage on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    X: numpy.ndarray shape (n, d) containing the dataset
    dist: maximum cophenetic distance for all clusters
    Displays the dendogram with each cluster displayed in a different color
    Returns clss, a numpy.ndarray of shape (n,) containing the cluster indeces
        for each data point
    """
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')

    dn = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist,
                                            above_threshold_color='b')
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, dist, 'distance')
    return clss
