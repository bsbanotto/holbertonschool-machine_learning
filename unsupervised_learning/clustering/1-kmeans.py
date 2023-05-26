#!/usr/bin/env python3
"""
Perform K-means clustering on a dataset
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

    if k <= 0:
        return None

    min_val = X.min(axis=0)
    max_val = X.max(axis=0)

    centroids = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))

    return centroids



def kmeans(X, k, iterations=1000):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
        n: number of datapoints
        d: number of dimensions for each datapoint
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
        that should be performed
    If no change in the cluster between iterations, function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    Should use numpy.random.uniform exactly twice
    May use at most 2 loops
    Returns C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n, ) containin the index of the cluster C
            that each data point belongs to
    """
    n = X.shape[0]
    d = X.shape[1]
    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    
    for i in range(iterations):
        # Calculate distance between centroids and data points
        delta = (X - centroids[:, None, :])  # (k, n, d)
        dist = np.linalg.norm(delta, axis=2).T  # (n, k)
        # Seperate into clusters
        clss = np.argmin(dist, axis=1)
        labeled = np.concatenate((X.copy(), np.reshape(clss, (n, 1))), axis=1)

        # Calculate the means of each cluster
        means = np.zeros((k, d))
        for j in range(k):
            temp = labeled[labeled[:, -1] == j]
            temp = temp[:, :d]
            if temp.size == 0:
                # new_centroid = np.random.uniform(low, high, size=(1, d))
                means[j] = initialize(X, 1)
            else:
                means[j] = np.mean(temp, axis=0)
        # Recalculate clss
        clss = np.argmin(np.linalg.norm((X - means[:, None, :]), axis=2).T,
                         axis=1)

        # Check for change
        if np.array_equal(centroids, means):
            break

        # Assign new centroids
        centroids = means

    return means, clss
