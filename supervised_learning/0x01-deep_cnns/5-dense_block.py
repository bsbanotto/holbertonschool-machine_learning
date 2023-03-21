#!/usr/bin/env python3
"""
Write a function that builds a densse block as described in
    `Densely Connectec Convolutional Networks`
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X: output from the previous layer
    nb_filters: integer representing the number of filters in X
    growth_rate: growth rate for the dense block
    layers: number of layers in the dense block
    Use the bottleneck layers used for DenseNet-B
    All weights should use the he normal initialization
    Convolutions should be preceded by Batch Normalization & ReLU activation
    Returns the concatenated output of each layer within the Dense Block and
        the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    concatenation = X
    for layer in range(layers):
        batch_norm_0 = K.layers.BatchNormalization(axis=3)(concatenation)
        ReLU_0 = K.layers.ReLU()(batch_norm_0)
        conv_0 = K.layers.Conv2D(filters=(4 * growth_rate),
                                 kernel_size=(1, 1),
                                 padding='same',
                                 kernel_initializer=init
                                 )(ReLU_0)
        batch_norm_1 = K.layers.BatchNormalization(axis=3)(conv_0)
        ReLU_1 = K.layers.ReLU()(batch_norm_1)
        conv_1 = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 kernel_initializer=init
                                 )(ReLU_1)
        concatenation = K.layers.Concatenate()([concatenation, conv_1])
        nb_filters = nb_filters + growth_rate

    return (concatenation, nb_filters)
