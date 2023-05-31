#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    # np.random.seed(11)
    # a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    # b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    # c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    # d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    # X = np.concatenate((a, b, c, d), axis=0)
    # np.random.shuffle(X)
    # pi, m, S = initialize(X, 4)
    # g, l = expectation(X, pi, m, S)
    # print(g)
    # print(g.shape)
    # print(np.sum(g, axis=0))
    # print(l)

    # np.random.seed(11)
    # a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    # b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    # c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    # d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    # X = np.concatenate((a, b, c, d), axis=0)
    # np.random.shuffle(X)
    # pi = np.array([0.75, 0.075, 0.075, 0.1])
    # m = np.array([[28, 42], [4, 23], [65, 31], [21, 65]])
    # S = np.array([[[70, 5], [5, 70]], [[15, 10], [10, 15]], [[15, 0], [0, 15]], [[30, 10], [10, 30]]])
    # g, l = expectation(X, pi, m, S)
    # print(g)
    # print(g.shape)
    # print(l)

    np.random.seed(1)
    m = np.random.randint(-100, 101, (3, 6))
    S = np.random.randint(-3, 3, (3, 6, 6))
    S = np.matmul(S, S.transpose(0, 2, 1))
    n = np.random.randint(100, 10001, 3)
    a = np.random.multivariate_normal(m[0], S[0], size=n[0])
    b = np.random.multivariate_normal(m[1], S[1], size=n[1])
    c = np.random.multivariate_normal(m[2], S[2], size=n[2])
    X = np.concatenate((a, b, c), axis=0)
    np.random.shuffle(X)
    i = np.random.randint(-3, 4, 3)
    pi = n + i
    pi = pi / np.sum(pi)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(g.shape)
    print(l)