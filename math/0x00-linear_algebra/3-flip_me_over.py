#!/usr/bin/env python3
"""
This file will transpose a matrix
"""


def matrix_transpose(matrix):
    """
    This function will transpose a matrix
    Column 1 will become row 1
    Column 2 will become row 2
    Column n will become row n
    """
    return list(map(list, zip(*matrix)))
