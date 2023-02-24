#!/usr/bin/env python3
"""
Function that calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    confusion: numpy.ndarray of shape (classes, classes)
        row: indices that represent the correct labels
        column: indices that represent the predicted labels
    Returns a numpy.ndarray of shape (classes, ) containing the specificity of
        each class
    """
    true_positives = np.diag(confusion)
    col_sum = np.sum(confusion, axis=0)
    false_positives = col_sum - true_positives
    row_sum = np.sum(confusion, axis=1)
    false_negatives = row_sum - true_positives
    subtraction = true_positives + false_positives + false_negatives
    true_negatives = np.sum(confusion) - subtraction
    speficity = true_negatives / (true_negatives + false_positives)
    return speficity
