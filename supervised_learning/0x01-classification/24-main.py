#!/usr/bin/env python3

import numpy as np
oh_encode = __import__('24-one_hot_encode').one_hot_encode

np.random.seed(2)
classes = np.random.randint(0, 20)
m = np.random.randint(100, 200)
Y = np.random.randint(0, classes, m).tolist()
np.set_printoptions(threshold=np.inf)
print(oh_encode(Y, classes))