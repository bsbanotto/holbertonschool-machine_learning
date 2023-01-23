#!/usr/bin/env python3
"""
This module defines the binomial distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Binomial:
    """
    This class is the Binomail distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a Binomial distribution
        """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if self.n <=0:
                raise ValueError("n must be a positive value")
            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        if type(data) is not list:
            raise TypeError("data must be a list")
        if len(data) <= 1:
            raise ValueError("data must contain multiple values")
