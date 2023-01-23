#!/usr/bin/env python3
"""
This module defines the exponential distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Exponential:
    """
    This class is the exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential class
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = len(data)/sum(data)

    def pdf(self, x):
        """
        Calculates the probability density function for time period 'x'
        if x is out of range(x <= 0), return 0
        pdf = lambtha * e^(-lambtha*x)
        """
        if x >= 0:
            return (self.lambtha * e ** ((-1 * self.lambtha) * x))
        else:
            return 0
