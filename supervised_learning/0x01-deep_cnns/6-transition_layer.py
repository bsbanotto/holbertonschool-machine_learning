#!/usr/bin/env python3
# Task 6. Transition Layer
"""
Write a function that builds a transition layer as described in
    `Densely Connected Convolutional Networks`
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X: the output from the previous layer
    nb_filters: integer representing the number of filters in X
    compression: compression factor for the transition layer
    Code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a ReLU
    Returns the output of the transition layer an the number of filters within
        the output, respectively
    """
    init = K.initializers.he_normal()
    filters = (int)(nb_filters * compression)

    batch_norm = K.layers.BatchNormalization(axis=3)(X)
    ReLU = K.layers.ReLU()(batch_norm)
    conv = K.layers.Conv2D(filters=filters,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=init
                           )(ReLU)
    AvgPool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'
                                        )(conv)

    return (AvgPool, filters)
