#!/usr/bin/env python3
"""
Function that calculates sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    confusion: numpy.ndarray of shape (classes, classes)
        row: indices that represent the correct labels
        column: indices that represent the predicted labels
    Returns a numpy.ndarray of shape (classes, ) containing the sensitivity of
        each class
    """
    true_positives = np.diag(confusion)
    row_sum = np.sum(confusion, axis=1)
    false_negative = row_sum - true_positives
    sensitivity = true_positives / (true_positives + false_negative)
    return sensitivity
