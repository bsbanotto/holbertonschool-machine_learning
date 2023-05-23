#!/usr/bin/env python3
import sys


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction')
if __name__ == '__main__':
    import numpy as np
    pca = __import__('1-pca').pca

    X = np.loadtxt("/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/data/mnist2500_X.txt")
    print('X:', X.shape)
    print(X)
    T = pca(X, 50)
    print('T:', T.shape)
    print(T)
