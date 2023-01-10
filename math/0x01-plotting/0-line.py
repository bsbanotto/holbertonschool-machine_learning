#!/usr/bin/env python3
"""
This module produces a line plot
"""
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
plt.plot(y, color='red')
plt.xlim([0, 10])
plt.show()
