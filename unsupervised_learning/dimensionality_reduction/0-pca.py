#!/usr/bin/env python3
"""
Function that performs Principal Component Analysis on a dataset that maintains
var fraction of X's original variance
"""
import numpy as np


def pca(X, var=0.95):
    """
    X: numpy.ndarray of shape (n, d) where:
        n: number of data points
        d: number of dimensions in each point
        all dimensions have a mean ov 0 across all data points
    var: fraction of the variance that the PCA transformation should maintain
    Returns the weights matrix, W, that maintains var fraction of X's original
        variance
    W: numpy.ndarray of shape (d, nd) where nd is the new dimensionality of the
        transformed X
    """
    # Perform singular value decomposition (SVD)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Calculate the explained variance ratio
    explained_variance_ratio = S**2 / np.sum(S**2)

    # Calculate the cumulative sum of explained variance ratio
    cumsum_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    # Find the number of dimensions needed to maintain the desired variance
    num_dimensions = np.argmax(cumsum_explained_variance_ratio >= var) + 1

    # Select the corresponding eigenvectors to form the weights matrix
    W = Vt[:num_dimensions + 1].T

    return W
