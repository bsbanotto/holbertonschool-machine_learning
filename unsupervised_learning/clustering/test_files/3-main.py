#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/clustering')
if __name__ == "__main__":
    optimum_k = __import__('3-optimum').optimum_k
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, kmax='5'))
    print(optimum_k(X, kmax=0))
    print(optimum_k(X, kmax=-1))