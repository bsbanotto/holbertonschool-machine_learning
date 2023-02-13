#!/usr/bin/env python3
"""
    x: the placeholder for the input data
    layer_sizes: list containing the number of nodes in each layer of the
    network
    activations: list containing the activation functions for each layer of the
    network
    Returns the prediction of the network in tensor form
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Calculates the forward propagation prediction of a neural network
    """
    prediction = x
    for node in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[node],
                                  activations[node])
    return prediction
