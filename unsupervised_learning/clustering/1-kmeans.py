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

    try:
        if k <= 0:
            return None

        min_val = X.min(axis=0)
        max_val = X.max(axis=0)

        centroids = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))

        return centroids
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
        n: number of datapoints
        d: number of dimensions fo' each datapoint
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
        that should be performed
    If no change in the cluster between iterations, function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    Should use numpy.random.uniform exactly twice
    May use at most 2 loops
    Returns C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d) containing the centroid means
        clss: numpy.ndarray of shape (n, ) containig the index of the cluster C
            that each data point belongs to
    """
    # Initialize our centroids
    centroids = initialize(X, k)

    # If guard all of the data
    if type(iterations) is not int or iterations <= 0:
        return None, None

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if centroids is None:
        return None, None

    n, d = X.shape

    for _ in range(iterations):
        # Assign each datapoint to a cluster
        clss = np.argmin((np.linalg.norm((X - centroids[:, None, :]),
                                         axis=2).T), axis=1)

        # Add labels to the dataset
        label = np.concatenate((X.copy(), np.reshape(clss, (n, 1))), axis=1)

        # Calculate the means of each cluster
        means = np.zeros((k, d))
        for j in range(k):
            # Create temp subset for j-th cluster
            temp = label[label[:, -1] == j]
            # Trim the cluster number from the temp subset
            temp = temp[:, :d]
            if temp.size == 0:
                # If a cluster has no points, pick some random data point as a
                # new centroid
                means[j] = initialize(X, 1)
            else:
                # Otherwise, calculate the mean value of a cluster
                means[j] = np.mean(temp, axis=0)
        # Recalculate clss
        clss = np.argmin(np.linalg.norm((X - means[:, None, :]), axis=2).T,
                         axis=1)

        # Check if change
        if np.array_equal(centroids, means):
            break

        # Assign new centroids
        centroids = means

    # print(delta.shape)
    return means, clss
