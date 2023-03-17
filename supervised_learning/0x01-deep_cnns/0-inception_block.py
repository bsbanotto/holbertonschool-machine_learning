#!/usr/bin/env python3
"""
Write a function that builds an inception block as described in Going Deeper
with Convolutions (2014)
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev: output from the previous layer
    filters: tuple containing F1, F3R, F3, F5R, F5, FPP
        F1: number of filters in the 1x1 convolution
        F3R: number of filters in the 1x1 convolution before the 3x3
        convolution
        F3: number of filters in the 3x3 convolution
        F5R: number of filters in the 1x1 convolution before the 5x5
        convolution
        F5: number of filters in the 5x5 convolution
        FPP: number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block should use a ReLU activation
    Returns: the concatenated output of the inception block
    """
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filters[4]
    FPP = filters[5]

    conv_1x1 = K.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu'
                               )(A_prev)

    conv_1x1_3x3 = K.layers.Conv2D(filters=F3R,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   activation='relu'
                                   )(A_prev)

    conv_3x3 = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'
                               )(conv_1x1_3x3)

    conv_1x1_5x5 = K.layers.Conv2D(filters=F5R,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   activation='relu'
                                   )(A_prev)

    conv_5x5 = K.layers.Conv2D(filters=F5,
                               kernel_size=(5, 5),
                               padding='same',
                               activation='relu'
                               )(conv_1x1_5x5)

    max_pool_3x3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                         strides=1,
                                         padding='same'
                                         )(A_prev)

    conv_1x1_pooled = K.layers.Conv2D(filters=FPP,
                                      kernel_size=(1, 1),
                                      padding='same',
                                      activation='relu'
                                      )(max_pool_3x3)

    output = K.layers.Concatenate()([
        conv_1x1, conv_3x3, conv_5x5, conv_1x1_pooled
    ])

    return output
