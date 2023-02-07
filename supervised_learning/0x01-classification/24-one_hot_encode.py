#!/usr/bin/env python3
"""
Module to implement one-hot encoding
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Returns a one-hot encoding of Y with shape(classes, m)
    where m is the number of examples
    """
    m = Y.shape[0]
    try:
        matrix = np.zeros([classes, m])
        matrix[Y, np.arange(m)] = 1
        return(matrix)
    except Exception:
        return None