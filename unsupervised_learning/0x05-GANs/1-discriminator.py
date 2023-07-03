#!/usr/bin/env python3
"""
Create Discriminator subclass
"""
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator subclass for the discriminator side of the network
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):
        """
                Initialize Discriminator class

        Args:
            input_size - integer size of the input tensor
            hidden_size - integer size of the hidden layer
            output_size - integer size of the output layer
        """
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the Discriminator network

        Args:
            x - Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.layers(x)
