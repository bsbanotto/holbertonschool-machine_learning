#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    EM = __import__('8-EM').expectation_maximization
    # np.random.seed(11)
    # a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    # b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    # c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    # d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    # X = np.concatenate((a, b, c, d), axis=0)
    # np.random.shuffle(X)
    # k = 4
    # pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    # clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    # plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    # plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    # plt.show()
    # print(X.shape[0] * pi)
    # print(m)
    # print(S)
    # print(l)
    
    # np.random.seed(11)
    # a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    # b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    # c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    # d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    # X = np.concatenate((a, b, c, d), axis=0)
    # np.random.shuffle(X)
    # k = 4
    # pi, m, S, g, l = EM(X, k, 100, verbose=True)
    # print(pi)
    # print(m)
    # print(S)
    # print(g)
    # print(l.round(5))

    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = EM(X, k, iterations=50, tol=1e-6, verbose=True)
    print(pi)
    print(m)
    print(S)
    print(g)
    print(l.round(6))