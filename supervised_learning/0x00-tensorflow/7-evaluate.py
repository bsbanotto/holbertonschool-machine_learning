#!/usr/bin/env python3
"""
Evaluates the output of the neural network
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X: numpy.ndarray containing the input data
    Y: numpy.ndarray containing the one-hot labels for X
    save_path: location to load the model from
    Returns the network's prediction, accuracy and loss
    """
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(save_path + '.meta')
    new_saver.restore(sess, save_path)
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    y_pred = tf.get_collection('y_pred')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    results = sess.run([y_pred, loss, accuracy], feed_dict={x: X, y: Y})
    return results[0], results[2], results[1]
