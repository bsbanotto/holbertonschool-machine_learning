#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    P: square 2D numpy.ndarray of shape (n, n) representing the standard
        transition matrix
        P[i, j]: probability of transitioning from state i to j
        n: number of states in the markov chain
    Returns True if it is absorbing, or False on failure
    """
    n = P.shape[0]

    if type(P) is not np.ndarray:
        return False

    if P.shape[1] != n:
        return False

    # If all values on diagonal are 1 it is absorbing (aka identity matrix)
    diag = np.diag(P)
    if (diag == 1).all():
        return True

    # If no value on diagonal is 1, it can't be absorbing
    if not (diag == 1).any():
        return False

    # Knowing that P is in standard form [[I, 0], [R, Q]], get each of these
    # Find length of diagonal 1's
    I_size = 0
    for i in range(0, len(diag)):
        if diag[i] == 1:
            I_size += 1
    I = np.eye(I_size)
    R = P[I_size:, :I_size]
    Q = P[I_size:, I_size:]

    # Get funcamental matrix (F = (I - Q)^-1)
    F = np.linalg.inv(I - Q)

    # Calculate F times R
    FR = np.matmul(F, R)

    if (FR == 0).all():
        return False
    return True
