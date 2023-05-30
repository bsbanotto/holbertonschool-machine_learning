#!/usr/bin/env python3
"""
Calculate the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    X: numpy.ndarray shape (n, d) containing the data points whose PDF should
        be evaluated
    m: numpy.ndarray shape (d,) containing the mean of the distribution
    S: numpy.ndarray shape (d, d) containing the covariance of the distribution
    Returns: P or None on failure
        P: numpy.ndarray of shape (n,) containing the PDF values for each
            data point
        All values in P should have a minimum value of 1e-300
    """
    # If guard against incorrect arguments
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    if X.shape[1] != m.shape[0]:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    if m.shape[0] != S.shape[0]:
        return None

    _, d = X.shape

    # Calculate the inverse and determinant of the covariance matrix
    inv_S = np.linalg.inv(S)
    det_S = np.linalg.det(S)

    if det_S <= 0:
        return None

    X_minus_m = X - m

    # Calculate the exponent of the PDF formula
    exponent = -0.5 * np.sum(X_minus_m @ inv_S * X_minus_m, axis=1)
    # Calculate the coefficient of the PDF formula
    coeff = 1 / np.sqrt((2 * np.pi) ** d * det_S)

    # Calculate the PDF values, and set mins to 1e-300
    P = coeff * np.exp(exponent)
    P = np.maximum(P, 1e-300)

    return P
