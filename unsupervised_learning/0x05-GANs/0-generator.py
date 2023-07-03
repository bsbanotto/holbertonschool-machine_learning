#!/usr/bin/env python3
"""
Create and initialize a generator subclass
"""
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator subclass for the generator side of the network
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):
        """
        Initialize Generator class

        Args:
            input_size - integer size of the input tensor
            hidden_size - integer size of the hidden layer
            output_size - integer size of the output layer
        """
        super(Generator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        def forward(self, x):
            """
            Forward pass of the Generator network

            Args:
                x - Input tensor of shape (batch_size, input_size)

            Returns:
                Output tensor of shape (batch_size, output_size)
            """
            return self.layers(x)
