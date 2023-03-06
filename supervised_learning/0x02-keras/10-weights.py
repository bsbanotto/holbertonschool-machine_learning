#!/usr/bin/env python3
"""
Write functions to save and load a models weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    network: model whose weights should be saved
    filename: path where the weights should be saved to
    save_format: format to which the weights should be saved
    Returns None
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """
    network: model whose weights should be loaded
    filename: path to the file that the weights shoule be loaded from
    Returns None
    """
    network.load_weights(filename)
