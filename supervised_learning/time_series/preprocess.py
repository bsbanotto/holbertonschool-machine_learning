#!/usr/bin/env python3
"""
This module will perform preprocessing on the coinbase dataset
Data cleaning, normalization, and splitting to train/val/test dataset
"""
from datetime import datetime
import pandas as pd


def preprocess():
    """
    Performs data preprocessing using pandas
    Args:
        None: Hard coded path to data

    Returns:
        train_df: Dataframe containing normalized and clean training data - 70%
        val_df: Dataframe containing normalized and clean validation data - 20%
        test_df: Dataframe containing normalized and clean test data - 10%
    """
    path = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

    # Convert file to pandas dataframe
    try:
        coinbase_df = pd.read_csv(path)
    except Exception:
        raise ValueError("Incorrect file path or File does not exist.")

    # Assign date as starting point for data set
    timestamp = datetime.timestamp(datetime.strptime('2017-08-20', "%Y-%m-%d"))

    # Drop all rows of the data frame prior to this date
    coinbase_df = coinbase_df[coinbase_df['Timestamp'] >= timestamp]

    # Interpolate over missing values and keep every 60th minute
    df = coinbase_df.interpolate()[::60]

    # split the data for training, validation, and testing
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # normalize the data based on training data
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
