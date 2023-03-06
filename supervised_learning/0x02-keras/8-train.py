#!/usr/bin/env python3
"""
Based on 7. Learning Rate Decay, update the function def train_model to also
save the best iteration of the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    save_best: boolean indicating whether to save the model after each epoch
    if it is the best
        A model is considered the best if its validation loss is the lowest
        that the model has obtained
    file_path: the file path to where the model should be saved
    learning_rate_decay: boolean that indicates whether or not learning rate
    decay should be used
        learning rate decay should only be performed if validation_data exists
        the decay should be performed using inverse time decay
        the learning rate should decay in a stepwise fashion after each epoch
        each time the learning rate updates, Keras should print a message
    early_stopping: boolean that indicates whether or not to stop early
        early stopping should only be performed if `validation_data` exists
        early stopping should be based on validation loss
    patience: patience used for early stopping
    validation_data: the data to validate the model with, if not `None`
    network: model to train
    data: numpy.ndarray of shape (m, nx) containing the input data
        m: number of data points
        nx: number of features
    labels: one-hot numpy.ndarray of shape (m, classes) with the data labels
        m: number of data points
        classes: number of classes
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data for mini-batch gradient descent
    verbose: boolean that determines if the output should be printed during
        training
    shuffle: boolean that determines whether to shuffle the batches every
        epoch. Normally, it is a good idea to shuffle, but for reproducibility
        we have chosen to set the default to False
    Returns: the History object generated after training the model
    """
    callback_list = []

    earlystopping = early_stopping
    if earlystopping and validation_data is not None:
        earlystopping = K.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=patience)
        callback_list.append(earlystopping)

    learningratedecay = learning_rate_decay
    if learningratedecay and validation_data is not None:

        def scheduler(epoch):
            return (alpha / (1 + decay_rate * epoch))
        learningratedecay = K.callbacks.LearningRateScheduler(scheduler,
                                                              verbose=1)
        callback_list.append(learningratedecay)

    savebest = save_best
    if savebest:
        savebest = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callback_list.append(savebest)

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callback_list
                       )
