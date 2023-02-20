#!/usr/bin/env python3
"""
Function that creates a learning rate decay operation in tensorflow using
inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow using
    inverse time decay
    alpha: original learning rate
    decay_rate: weight used to determine the rate at which alpha will decay
    global_step: the number of passes of gradient descent that have elapsed
    decay_step: the number of passes of gradient descent that should occur
    before alpha is decayed further
    The learning rate should occur in a stepwise fashion
    Returns the learning rate decay option
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
