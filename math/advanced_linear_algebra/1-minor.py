#!/usr/bin/env python3
"""
A function to calculate the minor matrix of a given matrix
"""


def zeros_matrix(rows, cols):
    """
    Creates a matrix of size rows x cols filled with zeros
    """
    zero_matrix = []
    while len(zero_matrix) < rows:
        zero_matrix.append([])
        while len(zero_matrix[-1]) < cols:
            zero_matrix[-1].append(0)

    return zero_matrix


def copy_matrix(matrix):
    """
    Makes a copy of a given matrix
    """
    rows = len(matrix)
    cols = len(matrix[0])

    matrix_copy = zeros_matrix(rows, cols)

    for row in range(rows):
        for col in range(cols):
            matrix_copy[row][col] = matrix[row][col]

    return matrix_copy


def determinant_recursive(matrix, determinant=0):
    """
    Recursively calculates the determinant of a given matrix
    """
    if matrix == [[]]:
        return 1

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])

    for col in range(len(matrix)):
        sub_mat = copy_matrix(matrix)
        sub_mat = sub_mat[1:]
        sub_height = len(sub_mat)

        for row in range(sub_height):
            sub_mat[row] = sub_mat[row][:col] + sub_mat[row][col + 1:]
        sign = (-1) ** (col % 2)
        sub_matrix_determinant = determinant_recursive(sub_mat)
        determinant += sign * matrix[0][col] * sub_matrix_determinant

    return determinant


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
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    cols = len(matrix[0])
    minor_matrix = zeros_matrix(rows, cols)

    for row in range(rows):
        for col in range(cols):
            sub_matrix = [row[:col] + row[col + 1:] for
                          row in matrix[:row] + matrix[row+1:]]
            sub_matrix_determinant = determinant_recursive(sub_matrix)
            minor_matrix[row][col] = sub_matrix_determinant

    return minor_matrix
