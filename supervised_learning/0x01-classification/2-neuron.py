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
        Initialize Neuron class, privatizing W, b, A
        """
        self.__b = 0
        self.__A = 0
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif(nx < 1):
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.randn(1, nx)

    @property
    def A(self):
        """
        Gets the Activation value
        """
        return self.__A

    @property
    def W(self):
        """
        Gets the Weight value
        """
        return self.__W

    @property
    def b(self):
        """
        Gets the bias value
        """
        return self.__b

    def forward_prop(self, X):
        """
        Calculates the forward propogation of the neuron
        """
        self.__A = (1 / (1 + np.exp(-X)))
        return self.__A
