#!/usr/bin/env python3
"""
prev: the tensor output of the previous layer
n: number of nodes in the layer to create
activation: activation function to use
Returns the tensor output of the layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer in our neural network
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=weights)
