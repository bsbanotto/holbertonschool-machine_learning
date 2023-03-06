#!/usr/bin/env python3
"""
Based on 6. Early Stopping, update the function def train_model to also train
model with learning rate decay
"""
import tensorflow.keras as K


def train_model_7(network, data, labels, batch_size, epochs,
                  validation_data=None, early_stopping=False, patience=0,
                  learning_rate_decay=False, alpha=0.1, decay_rate=1,
                  verbose=True, shuffle=False):
    """
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
    if validation_data is not None:
        if early_stopping and not learning_rate_decay:
            """Create earlystop callback"""
            earlystopping = K.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience)
            return network.fit(x=data,
                               y=labels,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose,
                               shuffle=shuffle,
                               validation_data=validation_data,
                               callbacks=[earlystopping]
                               )
        if learning_rate_decay and not early_stopping:
            """
            Create learning rate decay callback
            """
            def scheduler(epoch):
                """
                This function changes the learning rate in a stepwise manner
                using inverse time decay
                """
                return alpha / (1 + decay_rate * (epoch))
            learningratedecay = K.callbacks.LearningRateScheduler(scheduler,
                                                                  verbose)
            return network.fit(x=data,
                               y=labels,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose,
                               shuffle=shuffle,
                               validation_data=validation_data,
                               callbacks=[learningratedecay]
                               )
        if early_stopping and learning_rate_decay:
            earlystopping = K.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience)

            def scheduler(epoch):
                """
                This function changes the learning rate in a stepwise manner
                using inverse time decay
                """
                return alpha / (1 + decay_rate * (epoch))
            learningratedecay = K.callbacks.LearningRateScheduler(scheduler,
                                                                  verbose)
            return network.fit(x=data,
                               y=labels,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose,
                               shuffle=shuffle,
                               validation_data=validation_data,
                               callbacks=[earlystopping, learningratedecay]
                               )
    else:
        return network.fit(x=data,
                           y=labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data,
                           )
