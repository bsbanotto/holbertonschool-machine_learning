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

    def forward_prop(self, X):
        """
        Calculates the forward propagation for the neural network
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid1 = (1 / (1 + np.exp(-z1)))
        self.__A1 = sigmoid1
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid2 = (1 / (1 + np.exp(-z2)))
        self.__A2 = sigmoid2
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost for our neural network
        special value for 1 to avoid divide by 0 errors
        """
        one = 1.0000001
        loss = Y * np.log(A) + (1 - Y) * np.log(one - A)
        cost = -(1 / A.shape[1]) * np.sum(loss)
        return cost
