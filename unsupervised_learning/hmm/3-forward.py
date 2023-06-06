#!/usr/bin/env python3
"""
Performs the forward algorithm for a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    In lay terms, and from our main file, Observation is 364 days of clothing
    that we see a person wear, but it's random. We're meant to determine the
    probability this sequence of clothing choices happens (P). Spoiler alert
    it's a really, really small chance.

    Observation: numpy.ndarray shape (T,) that contains the index of the
        observation
        T: number of observations
    Emission: numpy.ndarray shape  (N, M) containing the emission probability
        of a specific observation given a hidden state
        Emission[i, j]: The probability of observing j given hidden state i
        N: Number of hidden states
        M: number of all possible observations
    Transition: numpy.ndarray shape (N, N) containing the transition probs
        Transition[i, j] is the probability of transitioning from i to j
    Initial: numpy.ndarray shape (N, 1) containing the probability of starting
        in a particular hidden state

    Returns P, F, or None, None on failure
        P: likelihood of the observations given the model
        F: numpy.ndarray shape (N, T) containing the forward path probs
            F[i, j] the probability of being in hidden state i at time j given
                the previous observations
    """
    # Get number of observations (T), number of hidden states (N) and number of
    # possible observations (M)
    T = len(Observation)
    N = Emission.shape[0]

    # Check to make sure our inputs are all the correct shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    # Initialize the forward path probs (F) to zero
    F = np.zeros((N, T))

    # Perform the initialization
    F[:, 0] = (Initial.T * Emission[:, Observation[0]])

    # For each observation, for each hidden state, multiply forward probs by
    # transition probs, then sum those up to get the forward prop at that time
    for t in range(1, T):
        for i in range(N):
            F[i, t] = np.sum(Emission[i, Observation[  # Messy line break here
                t]] * Transition[:, i] * F[:, t - 1])

    # Compute the likelihood of observations
    P = np.sum(F[:, T - 1])

    return P, F
