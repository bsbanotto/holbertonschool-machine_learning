#!/usr/bin/env python3
"""
Creates the training operation for a neural network in tensor flow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm
    loss: loss of the network
    alpha: learning rate
    beta1: momentum weight
    Returns the momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
