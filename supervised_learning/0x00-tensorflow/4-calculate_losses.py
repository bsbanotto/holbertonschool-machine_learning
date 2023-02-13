#!/usr/bin/env python3
"""
Module to calculate the softmax cross-entropy loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network's predictions
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
