#!/usr/bin/env python3
"""
Calculates sum of i^2, I range 1 to n
"""


def summation_i_squared(n):
    """
    given 'n', sum of i^2 from 1 to n
    """
    if (n < 1):
        return None
    return sum(map(lambda n : n * n, range(1, n + 1)))
