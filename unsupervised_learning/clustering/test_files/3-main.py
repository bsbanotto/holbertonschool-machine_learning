#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    optimum_k = __import__('3-optimum').optimum_k
    print("0-main")
    means = np.random.uniform(0, 100, (3, 2))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(2), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(2), size=10)
    c = np.random.multivariate_normal(means[2], 10 * np.eye(2), size=10)
    X = np.concatenate((a, b, c), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    print(np.round(v, 5))

    print("1-main")
    np.random.seed(1)
    means = np.random.uniform(0, 100, (2, 6))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(6), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(6), size=10)
    X = np.concatenate((a, b), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    print(np.round(v, 5))

    print("2-main")
    np.random.seed(0)
    means = np.random.uniform(0, 100, (5, 2))
    a = np.random.multivariate_normal(means[0], 16 * np.eye(2), size=50)
    b = np.random.multivariate_normal(means[1], 16 * np.eye(2), size=50)
    c = np.random.multivariate_normal(means[2], 16 * np.eye(2), size=50)
    d = np.random.multivariate_normal(means[3], 16 * np.eye(2), size=50)
    e = np.random.multivariate_normal(means[4], 16 * np.eye(2), size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X, 2, 10)
    print(res)
    print(np.round(v, 5))

    print("3-main")
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3)).tolist()
    print(optimum_k(X))
    print(optimum_k(np.arange(100)))
    print(optimum_k(np.arange(100).reshape((100, 1, 1))))

    print("4-main")
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, kmin='2'))
    print(optimum_k(X, kmin=0))
    print(optimum_k(X, kmin=-1))

    print("5-main")
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, kmax='5'))
    print(optimum_k(X, kmax=0))
    print(optimum_k(X, kmax=-1))

    print("6-main")
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, kmin=3, kmax=2))
    print(optimum_k(X, kmin=2, kmax=2))

    print("7-main")
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, iterations='10'))
    print(optimum_k(X, iterations=0))
    print(optimum_k(X, iterations=-1))
