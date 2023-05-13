#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking to
find the probability that a patient who takes this drug will develop severe
side effects. During your trials, n patients take the drug and x patients
develop severe side effects. You can assume that x follows a binomial
distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
    x: number of patients that develop sever side effects
    n: total number of patients observed
    P: 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
    if n is not a positive integer, raise a ValueError with the message
        n must be a positive integer
    if x is not an integer that is greater than or equal to 0, raise ValueError
        x must be an integer that is greater than or equal to 0
    if x is greater than n, raise a ValueError with the message
        x cannot be greater than n
    if P is not a 1D numpy.ndarray, raise a TypeError with the message
        P must be a 1D numpy.ndarray
    if any value in P is not in the range [0, 1], raise a ValueError message
        All values in P must be in the range [0, 1]
    Returns a 1D numpy.ndarray containing the likelihood of obtaining the data
        x and n for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        e_msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(e_msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if ((P <= 0) & (P >= 1)).all():
        raise ValueError("All values in P must be in the range [0, 1]")
    return 0
