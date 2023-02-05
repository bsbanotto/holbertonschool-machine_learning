#!/usr/bin/env python3
"""
Neural Network Module
"""
import numpy as np


class NeuralNetwork():
    """
    Constructs a NeuralNetwork
    """
    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1 weights vector"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1 bias vector"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1 activated output"""
        return self.__A1
        
    @property
    def W2(self):
        """Getter for W2 weights vector"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2 bias vector"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2 activated output"""
        return self.__A2
