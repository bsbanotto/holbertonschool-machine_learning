#!/usr/bin/env python3
"""
This module contains the Poisson class
"""
pi = 3.1415926536
e = 2.7182818285


class Poisson:
    """
    The Poisson class represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a poisson distribution
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
                self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """
        Calculates the probablity mass function (pmf)
        ((lambtha^k)*(e^-lambtha))/(k!)
        """
        def factorial(number):
            """
            Returns the factorial of a given number
            """
            factorial = 1
            for i in range(1, number + 1):
                factorial = factorial * i
            return factorial

        self.k = int(k)
        if k < 0:
            return 0
        return ((self.lambtha ** self.k * e ** (self.lambtha * -1))
                / factorial(k))
