#!/usr/bin/env python3
"""
Calculate the F1 score of a confusion matrix
"""
import numpy as np


def f1_score(confusion):
    """
    confusion: numpy.ndarray of shape (classes, classes)
        row: indices that represent the correct labels
        column: indices that represent the predicted labels
    Returns a numpy.ndarray of shape (classes, ) containing the F1 score of
        each class
    """
    # prec = precision(confusion)
    # sens = sensitivity(confusion)
    # F1 = prec + sens
    # return F1 / 2

    true_positives = np.diag(confusion)
    col_sum = np.sum(confusion, axis=0)
    false_positives = col_sum - true_positives
    row_sum = np.sum(confusion, axis=1)
    false_negatives = row_sum - true_positives
    F1 = (2 * true_positives) / ((2 * true_positives)
                                 + false_positives
                                 + false_negatives)
    return F1
