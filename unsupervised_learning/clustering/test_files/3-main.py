#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    optimum_k = __import__('3-optimum').optimum_k
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print('\n')
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()

    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3)).tolist()
    print(optimum_k(X))
    print(optimum_k(np.arange(100)))
    print(optimum_k(np.arange(100).reshape((100, 1, 1))))