#!/usr/bin/env python3
"""
Create placeholder Tensors for nx and classes
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two tensor placeholders, x and y, for a neural network
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return(x, y)
