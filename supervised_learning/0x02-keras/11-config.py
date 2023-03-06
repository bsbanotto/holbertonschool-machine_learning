#!/usr/bin/env python3
"""
Write functions to save and load a models configuration in JSON format
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    network: model whose configuration should be saved
    filename: path of the file that the configuration should be saved to
    Returns None
    """
    with open(filename, "w") as config_file:
        config_file.write(network.to_json())


def load_config(filename):
    """
    filename: path of the file containing the model's configuration in JSON
    Returns the loaded model
    """
    with open(filename, "r") as config_file:
        network = config_file.read()
    return K.models.model_from_json(network)
