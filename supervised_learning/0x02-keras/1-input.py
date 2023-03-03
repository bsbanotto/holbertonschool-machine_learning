#!/usr/bin/env python3
"""
Write a function that builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Not allowed to use `Sequential` class
    Returns the keras model
    """
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(units=layers[0],
                       activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha)
                       )(inputs)
    for layer in range(1, len(layers)):
        dropout_layer = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(units=layers[layer],
                           activation=activations[layer],
                           kernel_regularizer=K.regularizers.l2(lambtha)
                           )(dropout_layer)

    model = K.Model(inputs=inputs, outputs=x)
    return model
