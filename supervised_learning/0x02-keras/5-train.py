#!/usr/bin/env python3
"""
Based on 4. Train, update the function def train_model to also analyze
validation data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    validation_data: the data to validate the model with, if not `None`
    network: model to train
    data: numpy.ndarray of shape (m, nx) containing the input data
        m: number of data points
        nx: number of features
    labels: one-hot numpy.ndarray of shape (m, classes) with the data labels
        m: number of data points
        classes: number of classes
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data for mini-batch gradient descent
    verbose: boolean that determines if the output should be printed during
        training
    shuffle: boolean that determines whether to shuffle the batches every
        epoch. Normally, it is a good idea to shuffle, but for reproducibility
        we have chosen to set the default to False
    Returns: the History object generated after training the model
    """
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data
                       )
