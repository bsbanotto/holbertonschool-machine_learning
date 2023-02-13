#!/usr/bin/env python3
"""
Module that creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss: the loss of the network's prediction
    alpha: learning rate
    Returns an opearation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
