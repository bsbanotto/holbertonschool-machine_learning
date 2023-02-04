#!/usr/bin/env python3
"""
Neuron module
"""
import numpy as np
import matplotlib.pyplot as plt


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
        Matrix Multiply Weights and given numpy array, add bias
        Pass this to the sigmoid function
        """
        x = np.matmul(self.__W, X) + self.__b
        sigmoid = (1 / (1 + np.exp(-x)))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of a model using logistic regression
        Loss applies to a specific training example
        Cost is the sum of the Losses
        """
        one = 1.0000001
        loss = Y * np.log(A) + (1 - Y) * np.log(one - A)
        cost = -(1 / A.shape[1]) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neurons predictions
        """
        return (np.rint(self.forward_prop(X)).astype(int),
                self.cost(Y, self.__A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass(iteration) of gradient descent on the neuron
        updates private weight and bias attributes
        It learned and changed something on it's own!!
        """
        m = Y.shape[1]
        dz = A - Y
        dW = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - alpha * (dW).T

        self.__b = self.__b - alpha * (db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train our lonely neuron over a default of 5000 iterations
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif (iterations <= 0):
            raise ValueError("iterations must be a positive integer")
        else:
            self.iterations = iterations

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif (alpha <= 0):
            raise ValueError("alpha must be positive")
        else:
            self.alpha = alpha

        graphx = []
        graphy = []
        for i in range(0, self.iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, self.__A)))
            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, self.__A)
                    graphy.append(current_cost)
                    graphx.append(i)
                plt.plot(graphx, graphy)
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        plt.show()
        return (self.evaluate(X, Y))
