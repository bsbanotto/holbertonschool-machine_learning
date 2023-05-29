#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance
    # np.random.seed(0)
    # a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    # b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    # c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    # d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    # e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    # X = np.concatenate((a, b, c, d, e), axis=0)
    # np.random.shuffle(X)

    # for k in range(1, 11):
    #     C, _ = kmeans(X, k)
    #     print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))
    X = np.random.randn(100, 3)
    print(variance(X, 'hello'))
    print(variance(X, np.array([1, 2, 3, 4, 5])))
    print(variance(X, np.array([[[1, 2, 3, 4, 5]]])))
    print(variance(X, np.random.randn(5, 6)))
