#!/usr/bin/env python3
"""
Function that trains a GAN
"""
import torch


def train_gan(learning_rate=1e-3,
              batch_size=512,
              steps=5000,
              discriminator_steps=20,
              generator_steps=20,
              generator_input=1,
              generator_hidden=5,
              generator_output=1,
              discriminator_input=1,
              discriminator_hidden=10,
              discriminator_output=1):
    """
    Learning rate = 1e-3
    Batch size = 512
    Number of Iterations = 5000
    Number of steps for descriminator and generator = 20
    Use both Discriminator and Generator classes inside of function
    Returns the fake distribution of the Generator of type torch.Tensor()
    """
    train_dis = __import__('3-train_discriminator').train_dis
    train_gen = __import__('4-train_generator').train_gen
    sample_Z = __import__('2-sample_Z').sample_Z
    Discriminator = __import__('1-discriminator').Discriminator
    Generator = __import__('0-generator').Generator
    generator = Generator(generator_input,
                          generator_hidden,
                          generator_output)
    discriminator = Discriminator(discriminator_input,
                                  discriminator_hidden,
                                  discriminator_output)
    generator_optimizer = torch.optim.SGD(generator.parameters(),
                                          learning_rate)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(),
                                              learning_rate)
    loss = torch.nn.BCELoss()
    for step in range(steps):
        if step % 1000 == 0:
            print("Reached step {}".format(step))
        train_dis(generator, discriminator, generator_input,
                  discriminator_input, batch_size, 1, discriminator_optimizer,
                  loss)
        train_gen(generator, discriminator, generator_input,
                  batch_size, 1, generator_optimizer, loss)

    return generator(sample_Z(mu=0, sigma=1, sampleType="G",
                              size=(batch_size, generator_input)))


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import plotly.express as px

    ganTrainer = __import__('5-train_GAN').train_gan

    fakeData = ganTrainer()

    values = fakeData.data.storage().tolist()
    fig = px.histogram(values, title="Histogram of Forged Distribution",
                       nbins=50)
    fig.show()
