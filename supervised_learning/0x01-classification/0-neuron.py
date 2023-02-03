#!/usr/bin/env python3
"""
Neuron module
"""
import numpy as np


class Neuron():
    """
    Constructs a Neuron
    """
    def __init__(self, nx):
        """
        Initialize Neuron class
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif(nx < 1):
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.randn(1, 784)
            self.b = 0
            self.A = 0
    