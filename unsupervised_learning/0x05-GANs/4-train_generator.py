#!/usr/bin/env python3
"""
Function to train the generator
"""
import torch


def train_gen(Gen,
              Dis,
              gInputSize,
              mbatchSize,
              steps,
              optimizer,
              crit):
    """
        Args:
            Gen, Dis - Generator and Discriminator objects
            gInputSize - Input size of Generator input data
            mbatchSize - Batch size for training
            steps - number of steps for training
            optimizer - stochastic gradient descent optimizer object
            crit - BCEloss function

        Should use only random noise for sampling
        The 4 moments should be used in processing the sample

        Returns:
            The function should return the error of the fake data and the fake
            data set of type torch.Tensor()
    """
    sample_Z = __import__('2-sample_Z').sample_Z
    for _ in range(steps):
        # Generate data, same as discriminator training
        generated = sample_Z(mu=0.0,
                             sigma=1.0,
                             sampleType="G",
                             size=(mbatchSize, gInputSize))
        generatedOutput = Gen(generated)
        labels = torch.ones((mbatchSize, 1))

        # Train the generator
        Gen.zero_grad()
        output = Dis(generatedOutput)
        loss = crit(output, labels)
        loss.backward()
        optimizer.step()

    return loss, generated
