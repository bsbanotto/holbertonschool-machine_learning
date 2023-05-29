#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    optimum_k = __import__('3-optimum').optimum_k
    np.random.seed(0)
    means = np.random.uniform(0, 100, (3, 2))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(2), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(2), size=10)
    c = np.random.multivariate_normal(means[2], 10 * np.eye(2), size=10)
    X = np.concatenate((a, b, c), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    print(np.round(v, 5))