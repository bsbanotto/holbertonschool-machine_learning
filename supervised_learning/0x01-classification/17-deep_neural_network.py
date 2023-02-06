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
