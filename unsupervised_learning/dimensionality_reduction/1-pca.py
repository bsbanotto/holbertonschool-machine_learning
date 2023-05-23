#!/usr/bin/env python3
"""
Function that performs Principal Component Analysis on a dataset given the
number of dimensions to maintain.
"""
import numpy as np


def pca(X, ndim):
    """
    X: numpy.ndarray of shape (n, d)
        n: number of data points
        d: number of dimensions in each data point
    ndim: new dimensionality of the transformed X
    Returns T, a numpy.ndarray of shape(n, ndim) containing the transformed
        version of X
    """
    # Normalize X
    X_norm = X - np.mean(X, axis=0)

    # Perform singular value decomposition (SVD)
    _, _, Vt = np.linalg.svd(X_norm, full_matrices=False)

    # Select the corresponding eigenvectors to form the weights matrix
    W = Vt[:ndim].T

    # Transform the original dataset using ndim Principal Components
    T = np.dot(X_norm, W)

    return T
