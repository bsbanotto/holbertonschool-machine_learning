#!/usr/bin/env python3
"""
Calculate the most likely sequence of hidden states for a hidden markov model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Here's a simple descirption of the Viterbi Algorithm
    Imagine you have a special toy with different buttons, and each button
    corresponds to a hidden state. When you press a button, the toy emits a
    sound or makes a movement, which represents an observation. The toy has a
    hidden pattern of how it switches between different states and emits
    different sounds.

    Now, let's say you want to figure out the most likely sequence of buttons
    pressed based on the sounds or movements you observed. This is where the
    Viterbi algorithm comes in!

    Args:
        Observation: numpy.ndarray shape (T,) that contains the index of the
            observation
            T: number of observations
        Emission: numpy.ndarray shape (N, M) containing the emission prob of a
            specific observation given a hidden state
            Emission[i, j]: the probability of observing j given hidden state i
            N: Number of hidden states
            M: number of all possible observations
        Traisition: numpy.ndarray shape (N, N) containing the transition probs
            Transition[i, j] is the probability of transitioning from hidden
                state i to j
        Initial: numpy.ndarray shape (N, 1) containing the probability of
            starting in a particular hidden state

    Returns:
        path, P or None, None on failure
            path: list of length T containing the most likely sequence of
                hidden states
            P: probability of obtaining the path sequence
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    # Check to make sure our inputs are all the correct shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    path = np.zeros(T)
    V = np.zeros((N, T))

    V[:, 0] = (Initial.T * Emission[:, Observation[0]])

    # Loop through Observations and Hidden States to calculate Viterbi path (V)
    for t in range(1, T):
        for n in range(N):
            temp = V[:, t - 1] * Transition[:, n]
            max_index = np.argmax(temp)
            V[n, t] = temp[max_index] * Emission[n, Observation[t]]
            path[t] = max_index

    P = np.max(V[:, T - 1])

    max_index = np.argmax(V[:, T - 1])
    path[T - 1] = max_index
    for t in range(T - 2, -1, -1):
        path[t] = np.argmax(V[:, t])
        break

    return path.astype(int), P
