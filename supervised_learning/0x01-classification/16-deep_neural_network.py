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
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layers = [nx] + layers
        for i, l in enumerate(layers[1:], start=1):
            self.weights["W{}".format(i)] = (
                np.random.randn(1, layers[i - 1]) *
                np.sqrt(2 / (layers[i - 1]))
            )
            self.weights["b{}".format(i)] = np.zeros((l, 1))
