#!/usr/bin/env python3
"""
Function that creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    labels: one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels
        m: number of data points
        classes: number of classes
    logits: one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels
    Returns a confusion numpy.ndarray of shape (classes, classes)
        row indices: correct labels
        column indixes: predicted labels
    """
    confusion_matrix = np.zeros([labels.shape[1], logits.shape[1]])
    # print(confusion_matrix)
    true_label_index = np.where(labels == 1)[1]
    true_logits_index = np.where(logits == 1)[1]
    # print(true_label_index)
    # print(true_logits_index)
    indexes = list(zip(true_label_index, true_logits_index))
    unique, counts = np.unique(indexes, return_counts=True, axis=0)
    # print(unique)
    # print(counts)
    confusion_matrix[unique[:, 0], unique[:, 1]] = counts
    # print(confusion_matrix)

    return(confusion_matrix)
