#!/usr/bin/env python3

import os
import numpy as np
Deep = __import__('26-deep_neural_network').DeepNeuralNetwork

np.random.seed(4)
nx, m = np.random.randint(100, 200, 2).tolist()
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
deep = Deep(nx, [3, 1])
deep.train(X, Y, iterations=10, graph=False, verbose=False)
deep.save('1-test')
print(os.path.exists('1-test.pkl'))
