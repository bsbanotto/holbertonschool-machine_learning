#!/usr/bin/env python3
"""
This file performs matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    This function does what is needed
    """
    return_array = np.dot(mat1, mat2)
    return return_array
