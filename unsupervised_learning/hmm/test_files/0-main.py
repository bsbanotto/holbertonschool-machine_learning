#!/usr/bin/env python3
import sys
import numpy as np

sys.path.insert(0, '/home/bsbanotto/holbertonschool-machine_learning/unsupervised_learning/hmm')
if __name__ == "__main__":
    markov_chain = __import__('0-markov_chain').markov_chain
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))