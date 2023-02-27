#!/usr/bin/env python3
"""
Function that calculates the cost of a neural network with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    cost: tensor containing the cost of the network without L2 reg
    Return a tensor containing the cost of the network accounting for L2 reg
    """
    return (cost + tf.losses.get_regularization_losses())
