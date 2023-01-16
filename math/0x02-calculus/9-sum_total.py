#!/usr/bin/env python3
"""
Calculates sum of i^2, I range 1 to n
"""


def summation_i_squared(n):
    """
    given 'n', sum of i^2 from 1 to n
    """
    return(sum(i * i for i in range(1, n + 1)))
