#!/usr/bin/env python3
"""
Create input for the generator and discriminator
"""
import torch


def sample_Z(mu, sigma, sampleType, size=(1, 1)):
    """
    Creates input for the generator and discriminator

    Added size argument to sample_Z function. Per torch.randn and torch.normal
        documentataion, size is a sequence of integers defining the shape of
        the output tensor.

    Args:
        mu -  The mean of the distribution
        sigma - The standard deviation of the distribution
        sampleType - A variable that selects which model to sample for
            The variable should accept a G or D as string values

    Returns:
        a torch.Tensor type for both generator and discriminator if the
            parameters are correct, otherwise return 0
    """
    # For the generator, find a random number from torch.rand
    # Multiply that number by sigma and add mu
    # From the task, input data for the generator should be random
    if sampleType == "G":
        return torch.rand(size)
    # For the discriminator, take a number from the normal distribution
    # using torch.normal
    elif sampleType == "D":
        return torch.normal(mu, sigma, size)
    else:
        return 0
