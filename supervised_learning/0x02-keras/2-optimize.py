#!/usr/bin/env python3
"""
Write a function that sets up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    network: the model to optimize
    alpha: learning rate
    beta1: the first Adam optimization parameter
    beta2: second Adam optimization parameter
    Returns none
    """
    Adam_opt = K.optimizers.Adam(lr=alpha,
                                 beta_1=beta1,
                                 beta_2=beta2)
    network.compile(optimizer=Adam_opt,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
    return None
