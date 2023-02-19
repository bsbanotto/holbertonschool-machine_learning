#!/usr/bin/env python3
"""
Moving average calculator
"""


def moving_average(data, beta):
    """
    Calculates the bias corrected weighted moving average of a data set
    data: list of data to calculate the moving average of
    beta: weight used for the moving average
    Returns a list containing the moving averages of data
    """
    weighted_avg = 0
    weighted_average_list = []
    for count in range(len(data)):
        weighted_avg = (weighted_avg * beta) + ((1 - beta) * data[count])
        bias_correction = weighted_avg / (1 - (beta ** (count + 1)))
        weighted_average_list.append(bias_correction)
    return weighted_average_list
