#!/usr/bin/env python3
"""
Write a function that builds a modified version of the LeNet-5 architecture
using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    x: tf.placeholder of shape (m, 28, 28, 1) containing the input images for
    the network
        m: number of images
    y: tf.placeholder of shape (m, 10) containing the one-hot labels for the
    network
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with `same` padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with `valid` padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should use the relu activation
    function
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    convolutional1 = tf.layers.conv2d(inputs=x,
                                      filters=6,
                                      kernel_size=(5, 5),
                                      padding='same',
                                      activation='relu',
                                      kernel_initializer=init)

    max_pooling1 = tf.layers.max_pooling2d(inputs=convolutional1,
                                           pool_size=(2, 2),
                                           strides=(2, 2))

    convolutional2 = tf.layers.conv2d(inputs=max_pooling1,
                                      filters=16,
                                      kernel_size=(5, 5),
                                      padding='valid',
                                      activation='relu',
                                      kernel_initializer=init)

    max_pooling2 = tf.layers.max_pooling2d(inputs=convolutional2,
                                           pool_size=(2, 2),
                                           strides=(2, 2))

    flat_pool = tf.layers.flatten(max_pooling2)

    fully_connected_1 = tf.layers.dense(inputs=flat_pool,
                                        units=120,
                                        activation='relu',
                                        kernel_initializer=init,
                                        )

    fully_connected_2 = tf.layers.dense(inputs=fully_connected_1,
                                        units=84,
                                        activation='relu',
                                        kernel_initializer=init,
                                        )

    fully_connected_3 = tf.layers.dense(inputs=fully_connected_2,
                                        units=10,
                                        kernel_initializer=init)

    softmax = tf.nn.softmax(fully_connected_3)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=fully_connected_3)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_max = tf.math.argmax(y, axis=1)
    y_pred_max = tf.math.argmax(fully_connected_3, axis=1)
    equality = tf.math.equal(y_max, y_pred_max)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return softmax, optimizer, loss, accuracy
