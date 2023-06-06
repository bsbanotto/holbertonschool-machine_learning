#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    P: square 2D numpy.ndarray of shape (n, n) representing the transition
        matrix
        P[i, j]: probability of transitioning from state i to state j
        n: number of states in the markov chain
    Returns: numpy.ndarray of shape (1, n) containing the steady state
        probabbilities, or None on failure
    """
    # Check that P is a square matrix
    if P.shape[0] != P.shape[1]:
        return None

    # Check if P is a valid transition matrix
    if np.any(P < 0):
        return None
    if np.any(P > 1):
        return None
    if np.any(np.abs(np.sum(P, axis=1) - 1) > 1e-6):
        return None
    if np.any(P == 0):
        return None

    for i in range (100):
        steady_state_vector = np.linalg.matrix_power(P, i)

    return steady_state_vector[0, np.newaxis]
