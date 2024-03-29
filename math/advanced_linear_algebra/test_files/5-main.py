#!/usr/bin/env python3
import sys


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/math/advanced_linear_algebra')
if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1), "\n")
    print(definiteness(mat2), "\n")
    print(definiteness(mat3), "\n")
    print(definiteness(mat4), "\n")
    print(definiteness(mat5), "\n")
    print(definiteness(mat6), "\n")
    print(definiteness(mat7), "\n")
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
