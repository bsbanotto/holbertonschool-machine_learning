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
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif(nx < 1):
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.randn(1, nx)
            self.b = 0
            self.A = 0


    @property
    def W(self):
        """
        Gets the Weight value
        """
        return self.__W

    @W.setter
    def W(self, value):
        """
        Sets the Weight value
        """
        self.__W = value

    @property
    def b(self):
        """
        Gets the Weight value
        """
        return self.__b

    @b.setter
    def b(self, value):
        """
        Sets the Weight value
        """
        self.__b = value

    @property
    def A(self):
        """
        Gets the Weight value
        """
        return self.__A

    @A.setter
    def A(self, value):
        """
        Sets the Weight value
        """
        self.__A = value