#!/usr/bin/env python3
"""
Function that creates a tensorflow layer that includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function that should be used on the layer
    lambtha: the L2 regularization parameter
    Returns the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)

    l2_regular_layer = tf.layers.Dense(n,
                                       activation=activation,
                                       kernel_initializer=init,
                                       kernel_regularizer=reg)
    return l2_regular_layer(prev)
