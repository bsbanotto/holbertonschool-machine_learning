#!/usr/bin/env python3
"""
Creates a batch normalization layer for a nerual network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a nerual network in tensorflow
    prev: activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function that is to be used on the output of the
    layer
    Use tf.layers.Dense as the base layer with kernal initializer
        tf.contrib.layers.variance_Scaling_initializer(mode="FAN_AVG")
    Layer should incorporate two trainable parameters, gamma and beta
        gamma(scale in tf doc): initialized as vector of 1s
        beta(offset in tf doc): initialized as vector of 0s
    epsilon: 1e-8 to avoid divide by zero errors
    Returns a tensor of the normalized activated output for the layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.dense(inputs=prev,
                             units=n,
                             activation=activation,
                             kernel_initializer=weights)
    mean, variance = tf.nn.moments(layers, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1/100000000
    return activation(tf.nn.batch_normalization(layers,
                                                mean,
                                                variance,
                                                beta,
                                                gamma,
                                                epsilon))
