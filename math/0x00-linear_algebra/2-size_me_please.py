#!/usr/bin/env python3
"""
This file will determine the shape of a matrix
"""


def matrix_shape(matrix):
    """
    This function calculates the shape of a matrix
    """
    try:
        return [len(matrix)] + matrix_shape(matrix[0])

    except Exception:
        return [len(matrix)]
