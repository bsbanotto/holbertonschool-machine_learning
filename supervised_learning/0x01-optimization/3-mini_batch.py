#!/usr/bin/env python3
"""
Module that contains a function that shuffles the data in two matrices
in the same way.
"""
import tensorflow as tf
import numpy as np


shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent
    X_train: numpy.ndarray shape (m, 784) containing training data
        m: numper of data points (i.e. number of images)
        784: numper of input features (i.e. pixels per image)
    Y-train: one-hot numpy.ndarray shape (m, 10) containing training labels
        10: number of classes the model should classify
    X_valid: numpy.ndarray containing validation data
    Y_valid: one-hot numpy.ndarray containing validation labels
    batch_size: number of data points in a batch
    epochs: number of times training should pass through the whole dataset
    load_path: path from where the model should be loaded
    save_path: path to where the model should be saved after training
    Returns: path where the model was saved
    """
    with tf.Session() as sess:
        prev_batch = tf.train.import_meta_graph(load_path + '.meta')
        prev_batch.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')
        mini_batch = len(X_train)//batch_size + 1

        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x: X_train,
                                                             y: Y_train})
            val_cost, val_accuracy = sess.run((loss, accuracy),
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_accuracy))
            if i < epochs:
                X_Shuffled, Y_Shuffled = shuffle_data(X_train, Y_train)
                for batch in range(mini_batch):
                    mini_batch_dict = {x: X_Shuffled[batch_size
                                                     * batch:batch_size
                                                     * (batch + 1)],
                                       y: Y_Shuffled[batch_size
                                                     * batch:batch_size
                                                     * (batch + 1)]}
                    sess.run((train_op), feed_dict=mini_batch_dict)
                    if batch % 100 == 0 and batch != 0:
                        mini_batch_cost = loss.eval(mini_batch_dict)
                        mini_batch_accuracy = accuracy.eval(mini_batch_dict)
                        print("\tStep {}:".format(batch))
                        print("\t\tCost: {}".format(mini_batch_cost))
                        print("\t\tAccuracy: {}".format(mini_batch_accuracy))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
