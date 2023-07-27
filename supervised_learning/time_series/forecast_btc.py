#!/usr/bin/env python3
"""
Creates, Trains, and Validates a RNN model for forecasting the price of BTC
Given a 24 hour window, predicts the price at the next hour
"""
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():
    """
    Creates a window for the time series
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, label_columns=None):
        """
        Initialize the Window Generator Class
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size
                                       )[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size
                                       )[self.labels_slice]

    def split_window(self, features):
        """
        Splits the generated window to inputs and label
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Converts dataframes to a keras dataset
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        Makes training dataset
        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """
        Makes validation datase
        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """
        Makes test dataset
        """
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model, window, patience=2, epochs=20):
    """
    Performs model.compile and model.fit on our RNN model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min'
        )

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


def forecast(train, val, test):
    """
    We will create and train our forecasting model
    Args:
        train: Training dataset
        val: Validation dataset
        test: Testing dataset
    """
    window = WindowGenerator(24, 1, 1, train, val, test,
                             ['Weighted_Price'])

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(24, return_sequences=True),
        tf.keras.layers.Dense(units=1)
        ])

    history = compile_and_fit(lstm_model, window)

    val_performance = {}
    performance = {}
    val_performance["LSTM"] = lstm_model.evaluate(window.val)
    performance["LSTM"] = lstm_model.evaluate(window.test, verbose=0)


if __name__ == "__main__":
    preprocess = __import__('preprocess').preprocess
    train_df, val_df, test_df = preprocess()
    forecast(train_df, val_df, test_df)
