#!/usr/bin/env python3
"""
Deep Neural Network Module
"""
import numpy as np


class DeepNeuralNetwork():
    """
    Deep Neural Network class
    """
    def __init__(self, nx, layers):
        """
        Constructs a Deep Neural Network
        Weights are initialized using He et al. method
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prevLayer = nx

        for l in range(len(layers)):
            w = np.random.randn(layers[l], prevLayer) * np.sqrt(2 / prevLayer)
            self.weights["W{}".format(1 + l)] = w
            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
            prevLayer = layers[l]

    @property
    def L(self):
        """
        getter for L, number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        getter for cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter for weights
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propogation values for our deep neural network
        X is the input data, shape of (nx, m)
        """
        A = X
        self.__cache["A{}".format(0)] = X
        for l in range(1, self.L + 1):
            W = self.weights["W{}".format(l)]
            b = self.weights["b{}".format(l)]
            z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache["A{}".format(l)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """
        Calculates the cost of our model using logistic regression
        Y contains the correct data labels
        A contains the activated output of the neuron for each example
        """
        one = 1.0000001
        loss = Y * np.log(A) + (1 - Y) * np.log(one - A)
        cost = -(1 / A.shape[1]) * np.sum(loss)
        return cost
