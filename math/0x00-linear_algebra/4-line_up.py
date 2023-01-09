#!/usr/bin/env python3
"""
This file will add two matrices
"""


def add_arrays(arr1, arr2):
    """
    This function adds two arrays
    If the arrays are of different length, return None
    """
    if (len(arr1) == len(arr2)):
        sum_array = []
        for i in range(len(arr1)):
            sum_array.append(arr1[i] + arr2[i])
        return sum_array
    else:
        return None
