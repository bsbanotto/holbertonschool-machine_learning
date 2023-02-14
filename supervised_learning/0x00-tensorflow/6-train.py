#!/usr/bin/env python3
"""
Method to tie together all of the tensorflow stuff we've been working on
"""
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Function to build, train, and save a neural network classifier
    X_train: numpy.ndarray containing the input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: list containing the number of nodes in each nn layer
    activations: list containing the activation functions for each nn layer
    alpha: learning rate
    iterations: number of iterations to train over
    save_path: designates where to save the model
    Returns the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for i in range(iterations + 1):
            if i % 100 == 0 or i == iterations:
                """Find cost and accuracy only on every 100th and last iter"""
                train_cost, train_accuracy = sess.run((loss, accuracy),
                                                      feed_dict={x: X_train,
                                                                 y: Y_train}
                                                      )
                val_cost, val_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x: X_valid,
                                                             y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(val_cost))
                print("\tValidation Accuracy: {}".format(val_accuracy))
            if i < iterations:
                """Train our network for all iterations"""
                sess.run((train_op), feed_dict={x: X_train, y: Y_train})
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
