#!/usr/bin/env python3
"""
Module that implements a one-hot decoding
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Function to decode a one-hot matrix into a vector of labels
    """
    onehot_decode = []
    try:
        for row in one_hot.T:
            for i in range(0, len(row)):
                if row[i] == 0:
                    i += 1
                if row[i] == 1:
                    break
            onehot_decode.append(i)
        return(np.asarray(onehot_decode))
    except Exception:
        return None
