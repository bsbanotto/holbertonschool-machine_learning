#!/usr/bin/env python3
"""
Function that determines if gradient descent should stop early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    cost: the current validation cost of the neural network
    opt_cost: the lowest recorded validation cost of the network
    threshold: the threshold used for early stopping
    patience: the patience count used for early stopping
    count: count of how long the threshold has not been met
    Returns a boolean of whether the network should be stopped early
        followed by the updated count
    """
    cost_gap = opt_cost - cost

    if cost_gap > threshold:
        # Reset count because cost_gap is larger than threshold
        count = 0
    else:
        # Add to count and wait patiently
        count += 1

    if count < patience:
        # We don't want to stop because we haven't waited long enough
        return(False, count)
    else:
        # We waited long enough and should stop early
        return(True, count)
