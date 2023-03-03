#!/usr/bin/env python3
"""
Runction that converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    labels: labels for dat
    """
    return K.utils.to_categorical(labels, classes)
