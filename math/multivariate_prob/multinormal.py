#!/usr/bin/env python3
"""
Class Multinormal that represents a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
    Class Multinormal that represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        class constructor:
        data: numpy.ndarray shape (d, n) containing the data set
            d: number of dimensions in each data point
            n: number of data points
        if data is not a  2D numpy.ndarray, raise a TypeError with message
            data must be a 2D numpy.ndarray
        if n is less than 2, raise a ValueError with the message
            data must contain multiple data points
        Set the public instance variables:
            mean: np.ndarray shape (d, 1) containing the mean of data
            cov: np.ndarray shape (d, d) containing the covariance matrix data
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot(data-self.mean, (data-self.mean).T)/(data.shape[1]-1)
        self.det = np.linalg.det(self.cov)
        self.inv = np.linalg.inv(self.cov)

    def pdf(self, x):
        """
        Calculates the PDF of a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != self.mean.shape:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        coef = 1 / np.sqrt((2 * np.pi)**d * self.det)
        exponent = -0.5 * np.dot((x - self.mean).T,
                                 np.dot(self.inv, (x - self.mean)))
        return '%.19f' % float(coef * np.exp(exponent)) - 0.0000000000000000004
