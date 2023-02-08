#!/usr/bin/env python3
"""
Deep Neural Network Module
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """
        Evaluates the neural networks predictions
        X is an array that contains the input data
        Y is an array that contains the correct labels for the data
        """
        predictions, cache = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        evaluation = np.rint(predictions).astype(int)
        return evaluation, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent in our deep neural network
        Y is an array with shape (1. m) that contains the correct labels
        cache is a dictionary containing the intermediary values of the network
        alpha is the learning rate
        """
        m = Y.shape[1]

        for layer in range(self.__L, 0, -1):
            A_layer = self.__cache["A{}".format(layer)]
            A_layer_prev = self.__cache["A{}".format(layer - 1)]

            if layer == self.__L:
                dz_layer = (A_layer - Y)
            else:
                dz_layer = dA_layer_prev * (A_layer * (1 - A_layer))

            dW_layer = np.matmul(dz_layer, A_layer_prev.T) / m
            db_layer = np.sum(dz_layer, axis=1, keepdims=True) / m

            W_layer = self.__weights["W{}".format(layer)]
            dA_layer_prev = np.matmul(W_layer.T, dz_layer)

            self.__weights["W{}".format(layer)] -= alpha * dW_layer
            self.__weights["b{}".format(layer)] -= alpha * db_layer

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Lets train our neural network.
        X is the input data
        Y is the correctly labeled data for the input data
        iterations is how many times we're going to train
        alpha is our learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        graphx = []
        graphy = []
        for i in range(0, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))
            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, A)
                    graphy.append(current_cost)
                    graphx.append(i)
                plt.plot(graphx, graphy)
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.show()
            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be in integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        # plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """
        Saves the instance of our DeepNeuralNetwork object in pickle format
        filename is the file to which the object will be saved
        If filename doesn't have .pkl, it will be added
        """
        # Check to see if filename ends in .pkl
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        try:
            fileObject = open(filename, 'wb')
            pickle.dump(self, fileObject)
            fileObject.close()
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        filename is the file to be loaded
        Returns loaded object, or None if filename doesn't exist
        """
        try:
            fileObject = open(filename, 'rb')
            deepneuralnetwork = pickle.load(fileObject)
            fileObject.close()
            return (deepneuralnetwork)
        except Exception:
            return None
