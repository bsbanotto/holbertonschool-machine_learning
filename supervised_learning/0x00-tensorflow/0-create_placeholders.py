#!/usr/bin/env python3
"""
Create placeholder Tensors for nx and classes

Use this import for testing locally
`import tensorflow.compat.v1 as tf`

Use this import when committing to github
`import tensorflow as tf`
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two tensor placeholders, x and y, for a neural network
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return(x, y)
