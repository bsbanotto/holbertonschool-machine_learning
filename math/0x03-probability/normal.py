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