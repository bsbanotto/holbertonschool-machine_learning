#!/usr/bin/env python3
"""
This file adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    This function adds two 2D matrices
    """
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)
    if(shape_mat1 == shape_mat2):
        return_matrix = []
        return_row = []
        for i in range(len(mat1)):
            for j in range(len(mat1[i])):
                return_row.append(mat1[i][j] + mat2[i][j])
            return_matrix.append(return_row)
            return_row = []
        return return_matrix
    else:
        return None


def matrix_shape(matrix):
    """
    This function calculates the shape of a matrix
    """
    try:
        return [len(matrix)] + matrix_shape(matrix[0])

    except Exception:
        return [len(matrix)]
