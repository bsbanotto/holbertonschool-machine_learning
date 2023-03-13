#!/usr/bin/env python3
"""
Write a function that builds a modified version of the LeNet-5 architecutre
using Keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    X: K.Input of shape (m, 28, 28, 1) containing the input images for the
    network
        m: number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with `same` padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with `valid` padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method
    All hidden layers requiring activation should use the relu activation
    function
    Returns a K.Model compiled to use Adam optimization and accuracy metrics
    """
    init = K.initializers.he_normal()

    convolutional1 = K.layers.Conv2D(filters=6,
                                     kernel_size=(5, 5),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer=init)(X)

    max_pooling1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(convolutional1)

    convolutional2 = K.layers.Conv2D(filters=16,
                                     kernel_size=(5, 5),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer=init)(max_pooling1)

    max_pooling2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(convolutional2)

    flat_pool = K.layers.Flatten()(max_pooling2)

    fully_connected_1 = K.layers.Dense(units=120,
                                       activation='relu',
                                       kernel_initializer=init,
                                       )(flat_pool)

    fully_connected_2 = K.layers.Dense(units=84,
                                       activation='relu',
                                       kernel_initializer=init,
                                       )(fully_connected_1)

    fully_connected_3 = K.layers.Dense(units=10,
                                       activation='softmax',
                                       kernel_initializer=init
                                       )(fully_connected_2)

    model = K.models.Model(X, fully_connected_3)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
