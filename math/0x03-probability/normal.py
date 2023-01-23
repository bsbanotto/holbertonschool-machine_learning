#!/usr/bin/env python3
"""
This module defines the normal distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """
    This class is the normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal class
        """
        variance = 0
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            for number in data:
                variance = variance + (number - self.mean) ** 2
            variance = variance / len(data)
            self.stddev = variance ** .5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        z = (x - mean) / stddev
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return ((z * self.stddev) + self.mean)

    def pdf(self, x):
        """
        Calculates the value of the probability density function (pdf) for x
        see this link for the formula 
        https://en.wikipedia.org/wiki/Normal_distribution
        """
        val1 = 1 / (self.stddev * ((2 * pi) ** .5))
        val2 = -.5 * ((x - self.mean) / self.stddev) ** 2
        return (val1 * e ** val2)
