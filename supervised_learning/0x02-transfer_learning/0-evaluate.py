#!/usr/bin/env python3
"""
Script to evaluate our models for 0x02. Transfer Learning project
"""
# Imports
import os
import tensorflow.keras as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODEL_PATH = 'cifar10.h5'
# Load CIFAR10 dataset
(x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()

# Data Preprocessing
def preprocess_data(X, Y):
    """
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR10 data
    Y: numpy.ndarray of shape (m,) containing the CIFAR10 data labels
    Returns X_p and Y_p
    """
    X_p = K.applications.efficientnet.preprocess_input(X, data_format=None)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p

# Pre-process the data
x_train_p, y_train_p = preprocess_data(x_train, y_train)
x_valid_p, y_valid_p = preprocess_data(x_valid, y_valid)

# Load and evaluate the model
load_model = K.models.load_model(MODEL_PATH)
load_model.evaluate(x_valid_p, y_valid_p, verbose=1)
