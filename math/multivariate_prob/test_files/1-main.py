#!/usr/bin/env python3
import sys


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/math/multivariate_prob')
if __name__ == '__main__':
    import numpy as np
    correlation = __import__('1-correlation').correlation

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)