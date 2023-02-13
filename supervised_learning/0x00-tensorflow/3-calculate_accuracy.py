#!/usr/bin/env python3
"""
This module calculates the accuracy of our neural network
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y: placeholder for the labels of the input data
    y_pred: a tensor containing the network's predictions
    Returns a tensor containing the decimal accuracy of the prediction
    """
    y_max = tf.math.argmax(y, axis=1)
    y_pred_max = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(y_max, y_pred_max)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
