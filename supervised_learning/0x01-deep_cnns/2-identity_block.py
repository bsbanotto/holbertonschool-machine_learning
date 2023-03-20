#!/usr/bin/env python3
"""
Write a function that builds an identity block as described in
    `Deep Residual Learning for Image Recognition(2015)`
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev: output from the previous layer
    filters: tuple or list containing F11, F3, F12 respectively
        F11: number of filters in the first 1x1 convolution
        F3: number of filters in the 3x3 convolution
        F12: numnber of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch normalization
        along the channels axis and a rectified linear activation (ReLU)
    All weights should use he normal initialization
    Returns the activated output of the identity block
    """
    F11 = filters[0]
    F3 = filters[1]
    F12 = filters[2]

    init = K.initializers.he_normal()

    conv1x1_0 = K.layers.Conv2D(filters=F11,
                                kernel_size=(1, 1),
                                strides=1,
                                padding='same',
                                kernel_initializer=init
                                )(A_prev)

    batch_norm_0 = K.layers.BatchNormalization(axis=3)(conv1x1_0)

    relu_0 = K.layers.ReLU()(batch_norm_0)

    conv3x3_1 = K.layers.Conv2D(filters=F3,
                                kernel_size=(3, 3),
                                strides=1,
                                padding='same',
                                kernel_initializer=init)(relu_0)

    batch_norm_1 = K.layers.BatchNormalization(axis=3)(conv3x3_1)

    relu_1 = K.layers.ReLU()(batch_norm_1)

    conv1x1_2 = K.layers.Conv2D(filters=F12,
                                kernel_size=(1, 1),
                                strides=1,
                                padding='same',
                                kernel_initializer=init)(relu_1)

    batch_norm_2 = K.layers.BatchNormalization(axis=3)(conv1x1_2)

    add_layer = K.layers.Add()([A_prev, batch_norm_2])

    relu_output = K.layers.ReLU()(add_layer)

    return relu_output