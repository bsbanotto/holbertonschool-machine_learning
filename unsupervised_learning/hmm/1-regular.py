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

    # Compute eigenvals and eignevectors and find eigval with largest magnitude
    eigval, eigvec = np.linalg.eig(P.T)
    max_idx = np.argmax(np.abs(eigval))
    dominant_eigval = eigval[max_idx]

    # If dominant eigval is 1, return none
    if np.abs(dominant_eigval - 1) > 1e-6:
        return None

    steady_state_vector = np.real(eigvec[:, max_idx])

    steady_state_vector /= np.sum(steady_state_vector)

    steady_state_vector = steady_state_vector.reshape(1, P.shape[0])

    return steady_state_vector
