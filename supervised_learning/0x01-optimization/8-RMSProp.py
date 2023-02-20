#!/usr/bin/env python3
"""
Creates the training operation for a neural network using RMSProp in tensorflow
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Implementation of RMSProp algo using tensorflow
    loss: Loss of the network
    alpha: Learning rate
    beta2: RMSProp weight
    epsilon: Small number to avoid divide by zero error
    Returns the RMSProp optimization operation
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     epsilon=epsilon).minimize(loss)
