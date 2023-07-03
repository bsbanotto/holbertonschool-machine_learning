#!/usr/bin/env python3
import torch
"""
Function to train the Discriminator
"""


def train_dis(Gen,
              Dis,
              dInputSize,
              gInputSize,
              mbatchSize,
              steps,
              optimizer,
              crit):
    """
    Args:
        Gen, Dis - Generator and Discriminator objects
        dInputSize - Input size of Discriminator input data
        gInputSize - Input size of Generator input data
        mbatchSize - Batch size for training
        steps - number of steps for training
        optimizer - stochastic gradient descent optimizer object
        crit - BCEloss function

    Should use both random noise and normal distribution for sampling
    The 4 moments should be used in processing the sample

    Returns:
        The function should return the error estimate of the fake and real data
        along with the fake and real data sets of type torch.tensor()
    """
    sample_Z = __import__('2-sample_Z').sample_Z
    for _ in range(steps):
        # Create real and generated data and labels to train discriminator

        # Use sample_Z function to make some real data points
        reals = sample_Z(mu=0.,
                         sigma=1.,
                         sampleType="D",
                         size=(mbatchSize, dInputSize))
        # Make labels for real data, equal to 1
        realLabels = torch.ones((mbatchSize, 1))

        # Use sample_Z function to make some generated data
        generated = sample_Z(mu=0.,
                             sigma=1.,
                             sampleType="G",
                             size=(mbatchSize, gInputSize))
        # Make labels for generated data, equal to 0
        generatedLabels = torch.zeros((mbatchSize, 1))

        # Combine real and generated data/labels to run through discriminator
        data = torch.cat((reals, generated))
        labels = torch.cat((realLabels, generatedLabels))

        # Train the Discriminator on combined data and labels
        output = Dis(data)
        loss = crit(output, labels)
        loss.backward()
        optimizer.step()

    discriminatorOut = Dis(data)
    discriminatorLoss = crit(discriminatorOut, labels)

    return discriminatorLoss, data
