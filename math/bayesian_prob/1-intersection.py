#!/usr/bin/env python3
"""
Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr)
that calculates the intersection of obtaining this data with the various
hypothetical probabilities
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    x: number of patients that develop sever side effects
    n: total number of patients observed
    P: 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
    Pr: 1D numpy.ndarray containing the prior beliefs of P
    if n is not a positive integer, raise a ValueError with the message
        n must be a positive integer
    if x is not an integer that is greater than or equal to 0, raise ValueError
        x must be an integer that is greater than or equal to 0
    if x is greater than n, raise a ValueError with the message
        x cannot be greater than n
    if P is not a 1D numpy.ndarray, raise a TypeError with the message
        P must be a 1D numpy.ndarray
    if Pr is not a numpy.ndarray with the same shape as P, raise a TypeError
        Pr mjust be a numpy.ndarray with the same shape as P
    if any value in P or Pr is not in the range [0, 1] raise TypeError
        All values in {P} must be in the range [0, 1]
            where {P} is the incorrect variable
    if Pr does not sum to 1, raise a ValueError
        Pr must sum to 1
    Returns a 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively
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
    for value in P:
        if value < 0 or value > 1:
            raise ValueError("All values in {P} must be in the range [0, 1]")
    for value in Pr:
        if value < 0 or value > 1:
            raise ValueError("All values in {Pr} must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    likelihoods = np.zeros_like(P)
    fact = np.math.factorial
    n_choose_x = fact(n) / (fact(x) * fact(n - x))
    likelihoods = n_choose_x * (P ** x) * ((1 - P) ** (n - x))

    intersections = likelihoods * Pr
    return intersections
