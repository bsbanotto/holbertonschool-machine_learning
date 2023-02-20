#!/usr/bin/env python3
"""
Creates training operation for a neural network in tensorflow using the
Adam optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates training operation for a neural network in tensorflow using the
    Adam optimization algorithm
    loss: loss of the network
    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid divide by zero error
    Returns the Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
