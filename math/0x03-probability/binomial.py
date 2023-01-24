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
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            mean = sum(data)/len(data)
            variance = 0
            for number in data:
                variance = variance + (number - mean) ** 2
            variance = variance / len(data)
            q = variance / mean
            p1 = 1 - q
            n1 = (sum(data) / p1) / len(data)
            self.n = int(round(n1))
            self.p = float(mean/self.n)

    def pmf(self, k):
        """
        Calculates the probability mass function
        (n|k) * p^k * (1-p)^(n-k))
        """
        self.k = int(k)
        if k <= 0:
            return 0

        def factorial(number):
            """
            Returns the factorial of a given number
            """
            factorial = 1
            for i in range(1, number + 1):
                factorial = factorial * i
            return factorial

        coeff = factorial(int(self.n)) / (factorial(int(self.k)) *
                                          factorial(int(self.n) - int(self.k)))

        exp = self.n - k
        val1 = self.p ** k
        val2 = (1-self.p) ** exp
        return (coeff * val1 * val2)
