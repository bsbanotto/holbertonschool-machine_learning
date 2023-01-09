#!/usr/bin/env python3
"""
This file concatenates two matrices along a give axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    This function concatenates two matrices along an axis
    Axis = 0 concats rows
    Axis = 1 concats columns
    """
    return_matrix = []
    if(axis == 0):
        for row in mat1:
            return_matrix.append(row.copy())
        for row in mat2:
            return_matrix.append(row.copy())
        return return_matrix
    if(axis == 1):
        for col in range(len(mat1)):
            return_matrix.append(mat1[col].copy() + mat2[col].copy())
        return(return_matrix)