#!/usr/bin/env python3
"""
Module that implements a one-hot decoding
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Function to decode a one-hot matrix into a vector of labels
    """
    try:
        return (one_hot.argmax(0))
    except Exception:
        return None
