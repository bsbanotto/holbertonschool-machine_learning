#!/usr/bin/env python3
"""
A function to calculate the minor matrix of a given matrix
"""


def minor(matrix):
    """
    matrix: list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
        matrix must be a list of lists
    If matrix is not square, or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    Returns the minor matrix of matrix
    """
    rows = len(matrix)
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if rows == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != rows or rows == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    return(0)