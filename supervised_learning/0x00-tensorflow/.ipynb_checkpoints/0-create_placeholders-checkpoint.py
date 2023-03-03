#!/usr/bin/env python3
"""
Documentation here, will improve later after testing that tensorflow is working
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shame=(None, classes), name='y')
    return(x, y)
