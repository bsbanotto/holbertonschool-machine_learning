#!/usr/bin/env python3
"""
Determines the probability of a markov chain being in a particular state
after a specified number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    P: square 2D numpy.ndarray of shape (n, n) representing the transition
        matrix
        P[i, j]: the probability of transitioning from state i to j
        n: number of states in the markov chain
    s: numpy.ndarray of shape (1, n) representing the probability of starting
        in each state
    t: number of iterations that the markov chain has been through
    Returns a numpy.ndarray of shape (1, n) representing the probability of
        being in a specific state after t iterations, or None on failure
    """
    # If guard bad inputs
    if type(P) is not np.ndarray or P.shape(0) != P.shape(1):
        return None
    if type(s) is not np.ndarray or s.shape(1) != P.shape(0):
        return None
    if type(t) is not int or t <= 0:
        return None

    # for every iteration, s = s * P
    for _ in range(0, t):
        s = np.matmul(s, P)
    return s
