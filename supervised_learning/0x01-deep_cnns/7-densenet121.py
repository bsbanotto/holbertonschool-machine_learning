#!/usr/bin/env python3
# Task 7. DenseNet-121
"""
Write a function that builds the DenseNet-121 architecture as described in
    `Densely Connected Convolutional Networks`
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate: growth rate
    compression: compression factor
    Assume all input data will have the shape (224, 224, 3)
    All convolutions should be preceded by BN-ReLU
    All weights should use the he normal initialization
    Returns the keras model
    """
    inputs = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    batch_norm_0 = K.layers.BatchNormalization(axis=3)(inputs)
    ReLU_0 = K.layers.Activations('relu')(batch_norm_0)
    conv_0 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=2,
                             padding='same',
                             kernel_initializer=init
                             )(ReLU_0)
    MaxPooling = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=2,
                                       padding='same'
                                       )(conv_0)

    dense_block_0, nb_filters = dense_block(X=MaxPooling,
                                            nb_filters=64,
                                            growth_rate=growth_rate,
                                            layers=6)
    transition_block_0, nb_filters = transition_layer(X=dense_block_0,
                                                      nb_filters=nb_filters,
                                                      compression=compression)
    dense_block_1, nb_filters = dense_block(X=transition_block_0,
                                            nb_filters=nb_filters,
                                            growth_rate=growth_rate,
                                            layers=12)
    transition_block_1, nb_filters = transition_layer(X=dense_block_1,
                                                      nb_filters=nb_filters,
                                                      compression=compression)
    dense_block_2, nb_filters = dense_block(X=transition_block_1,
                                            nb_filters=nb_filters,
                                            growth_rate=growth_rate,
                                            layers=24)
    transition_block_2, nb_filters = transition_layer(X=dense_block_2,
                                                      nb_filters=nb_filters,
                                                      compression=compression)
    dense_block_3, nb_filters = dense_block(X=transition_block_2,
                                            nb_filters=nb_filters,
                                            growth_rate=growth_rate,
                                            layers=16)

    global_average = K.layers.AveragePooling2D(pool_size=(7, 7),
                                               strides=1,
                                               )(dense_block_3)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax'
                             )(global_average)

    return K.Model(inputs=inputs, outputs=softmax)
