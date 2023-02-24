#!/usr/bin/env python3
"""
Function that calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    confusion: numpy.ndarray of shape (classes, classes)
        row: indices that represent the correct labels
        column: indices that represent the predicted labels
    Returns a numpy.ndarray of shape (classes, ) containing the precision of
        each class
    """
    true_positives = np.diag(confusion)
    col_sum = np.sum(confusion, axis=0)
    false_positives = col_sum - true_positives
    precision = true_positives / (true_positives + false_positives)
    return precision
