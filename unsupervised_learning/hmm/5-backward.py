#!/usr/bin/env python3
"""
Perform the backward algorithm for a hidden markov model
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    If we look back at the Viterbi algorithm that gave us the most likely
    sequence of buttons, the Backward Algorithm helps us figure out the
    probability of being in each hidden state at a particular time

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
        P, B or None, None on failure
            P: likelihood of the observations given the model
            B: numpy.ndarray shape (N, T) containing the backward probabilities
                B[i, j]: probability of generating the future observations from
                    hidden state i at time j
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    # Check to make sure our inputs are all the correct shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    return("Hello", "Hello")
