#!/usr/bin/env python3
"""
Write fnctions to save and load an entire model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    network: model to save
    filename: path of the file that the model should be saved to
    Returns None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    filename: path of the fale that the model should be loaded from
    Returns the loaded model
    """
    return K.models.load_model(filename)
