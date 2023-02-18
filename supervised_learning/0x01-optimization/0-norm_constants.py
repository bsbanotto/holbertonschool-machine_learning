#!/usr/bin/env python3
"""
Module that contains a function that calculates the normalization
constants of a matrix
"""
import tensorflow as tf
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix
    X: numpy.ndarray of shape (m, nx)
    m: number of data points
    nx: number of features
    Returns: mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return (mean, stdev)
