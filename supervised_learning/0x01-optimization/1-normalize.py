#!/usr/bin/env python3
"""
Module that contains a function that normalizes a matrix
"""
import tensorflow as tf
import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix
    X: numpy.ndarray matrix of shape (d, nx)
    d: number of data points(called m in task above)
    nx: number of features
    m: mean of all features(calculated in task 0)
    s: standard deviation of each feature (calculated in task 0)
    Returns: Normalized X matirx
    """
    return ((X - m) / s)
