#!/usr/bin/env python3
"""
Write a function that builds a neural network with the Keras library
Use Sequential model since that's in the task name
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Not allowed to use `Input` class
    Returns the keras model
    """
    model = K.Sequential()
    # Create our input layer
    model.add(K.layers.InputLayer(input_shape=(nx,)))
    # Create all of our hidden layers
    for layer in range(len(layers)):
        model.add(K.layers.Dense(units=layers[layer],
                                 activation=activations[layer],
                                 kernel_regularizer=K.regularizers.l2(lambtha)
                                 ))
        # Handle dropout on hidden layers
        if layer < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
