#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network model in tensorflow
"""
import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation, last, epsilon):
    """
    Creates a batch normalization for a neural network
    prev: activated output of the previous layer
    n, number of nodes in the layer to be created
    activation: activation function that is to be used on the output of the
        layer
    last: boolean operator. If last layer, return that tensor for that layer
    epsilon: small number to avoid divide by zero errors

    This is similar to task 14, except we needed to add check for last layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.dense(inputs=prev,
                             units=n,
                             kernel_initializer=init)
    if last is True:
        return layers
    mean, variance = tf.nn.moments(layers, axes=[0])
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    return activation(tf.nn.batch_normalization(layers,
                                                mean,
                                                variance,
                                                beta,
                                                gamma,
                                                epsilon))


def forward_prop(input, epsilon, layer_sizes=[], activations=[]):
    """
    Use tensorflow to calculate forward propagation of the neural network
    input: placeholder for input data
    epsilon: small number to avoid divide by zero errors
    layer_sizes: list containing the number of nodes for each layer of the nn
    activations: list containing the activation functions for each layer
    Returns: prediction of the network in tensor form
    """
    prediction = input
    last = False
    for node in range(len(layer_sizes)):
        if node == len(layer_sizes) - 1:
            last = True
        prediction = create_batch_norm_layer(prediction,
                                             layer_sizes[node],
                                             activations[node],
                                             last,
                                             epsilon)
    return prediction


def calculate_accuracy(labels, pred_labels):
    """Calculates accuracy of prediction"""
    labels_max = tf.math.argmax(labels, axis=1)
    labels_pred_max = tf.math.argmax(pred_labels, axis=1)
    equality = tf.math.equal(labels_max, labels_pred_max)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    X: First numpy.ndarray matrix of shape (m, nx) to be shuffled
    Y: Second numpy.ndarray matrix of shape (m, ny) to be shuffled
    m: number of data points
    nx/ny: number of features in X and Y respectively
    """
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow
        See README for details on variables
    Data_train: tuple containing the training inputs and training labels
    Data_valid: tuple containing the validation inputs and labels
    layers: list containing the number of nodes in each layer
    activation: list containing the activation functions for each layer
    alpha: learning rate
    beta1: weight for first moment of Adam Optimization
    beta2: weight for second moment of Adam Optimization
    epsilon: small number to avoid divide by zero errors
    decay_rate: decay rate for inverse time decay of the learning rate
    batch_size: number of data points that should be in each mini-batch
    epochs: number of times the training should pass through the whole dataset
    save_path: path where the model should be saved to
    Returns the path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    data = tf.placeholder(name='data', dtype=tf.float32,
                          shape=[None, X_train.shape[1]])
    labels = tf.placeholder(name='labels', dtype=tf.float32,
                            shape=[None, Y_train.shape[1]])
    tf.add_to_collection('data', data)
    tf.add_to_collection('labels', labels)
    pred_labels = forward_prop(data, epsilon, layers, activations)
    tf.add_to_collection('y_pred', pred_labels)
    accuracy = calculate_accuracy(labels, pred_labels)
    tf.add_to_collection('accuracy', accuracy)
    loss = tf.losses.softmax_cross_entropy(labels, pred_labels)
    tf.add_to_collection('loss', loss)
    global_step = tf.Variable(0, trainable=False)
    mini_batch_size = len(X_train) // batch_size
    while mini_batch_size % batch_size != 0:
        mini_batch_size += 1
    alpha = tf.train.inverse_time_decay(learning_rate=alpha,
                                        global_step=global_step,
                                        decay_steps=mini_batch_size,
                                        decay_rate=decay_rate,
                                        staircase=True)
    train_op = tf.train.AdamOptimizer(alpha,
                                      beta1,
                                      beta2,
                                      epsilon).minimize(loss,
                                                        global_step)
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            train_loss = loss.eval({data: X_train,
                                    labels: Y_train})
            train_accuracy = accuracy.eval({data: X_train,
                                            labels: Y_train})
            validation_loss = loss.eval({data: X_valid,
                                         labels: Y_valid})
            validation_accuracy = accuracy.eval({data: X_valid,
                                                 labels: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(validation_loss))
            print("\tValidation Accuracy: {}".format(validation_accuracy))
            if epoch == epochs:
                break
            data_shuffled, labels_shuffled = shuffle_data(X_train, Y_train)
            for batch in range(0, data_shuffled.shape[0], batch_size):
                mini_batch_dict = {
                    data: data_shuffled[batch:batch + batch_size],
                    labels: labels_shuffled[batch:batch + batch_size]
                    }
                sess.run((train_op), feed_dict=mini_batch_dict)
                iterator = batch // batch_size + 1
                if iterator % 100 == 0 and iterator != 0:
                    mini_batch_cost = loss.eval(mini_batch_dict)
                    mini_batch_accuracy = accuracy.eval(mini_batch_dict)
                    print("\tStep {}:".format(iterator))
                    print("\t\tCost: {}".format(mini_batch_cost))
                    print("\t\tAccuracy: {}".format(mini_batch_accuracy))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
