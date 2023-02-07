#!/usr/bin/env python3
"""
Module that implements a one-hot decoding
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Function to decode a one-hot matrix into a vector of labels
    """
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot.ndim != 2:
        return None
    return (one_hot.argmax(0))
