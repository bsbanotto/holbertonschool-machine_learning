#!/usr/bin/env python3
"""
Given a state, action matrix and an action, weight matrix
Return the policy matrix, shape state, weight
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes a policy using the given weight for the provided matrix

    Args:
        matrix: np.ndarray shape (state, action)
        weight: np.ndarray shape (action, weight)

    Returns:
        The policy computed using the given weight
        np.ndarray shape (state, weight)
    """
    dot_prod = matrix.dot(weight)
    exp = np.exp(dot_prod)
    policy = exp / np.sum(exp)
    return policy
