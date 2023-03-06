#!/usr/bin/env python3
"""
Write a function that makes a prediction using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    network: network model to make the prediction with
    data: imput data to make the prediction with
    verbose: boolean that determines if output should be printed during the
    prediction process
    Returns the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)
