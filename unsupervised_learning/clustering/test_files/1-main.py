#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    kmeans = __import__('1-kmeans').kmeans
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    for i in range(1, 6):
        C, clss = kmeans(X, 5, iterations=i)
        print(C)
        plt.scatter(X[:, 0], X[:, 1], s=10, marker=".", c=clss, alpha=0.2)
        plt.scatter(C[:, 0], C[:, 1], s=100, marker='*', c=list(range(5)))
        plt.show()