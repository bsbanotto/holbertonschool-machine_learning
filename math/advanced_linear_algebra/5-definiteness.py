#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    matrix: numpy.ndarray shape (n, n) whose definiteness should be calculated
    If matrix is not a numpy.ndarray, raise a TypeError with message
        matrix must be a numpy.ndarray
    If matrix is not valid, return None
    Return the respective strings
        Positive definite
        Positive semi-definite
        Negative semi-definite
        Negative definite
        Indefinite
    If matrix does not fit any of the above categories, return None
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
    if np.all(eigvals > 0):
        return "Positive definite"
    if np.all(eigvals >= 0):
        return "Positive semi-definite"
    if np.all(eigvals < 0):
        return "Negative definite"
    if np.all(eigvals <= 0):
        return "Negative semi-definite"
    return "Indefinite"
