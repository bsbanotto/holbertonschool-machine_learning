#!/usr/bin/env python3
"""
Compute the Monte-Carlo policy gradient based on a state and weight matrix
"""
import numpy as np
policy = __import__('policy').policy


def policy_gradient(state, weight):
    """
    Function that computes the Monte-Carlo policy gradient based on a state
        and a weight matrix

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Returns:
        The action and the gradieng(in this order)
    """
    MCPolicy = policy(state, weight)
    action = np.random.choice(len(MCPolicy[0]), p=MCPolicy[0])

    # Need to reshape the policy to build softmax, so we do that here
    s = MCPolicy.reshape(-1, 1)

    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]

    log_derivative = softmax / MCPolicy[0, action]

    grad = state.T.dot(log_derivative[None, :])

    return action, grad
