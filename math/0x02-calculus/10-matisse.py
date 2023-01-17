#!/usr/bin/env python3
"""
Returns the coefficients of the derivative of a give polynomial
"""


def poly_derivative(poly):
    """
    poly is a list of coefficients representing a polynomial
    return the derivative
    """
    if len(poly) == 1:
        return 0
    if(type(poly) == list) or len(poly) > 0:
        return_list = []
        for i in range(1, len(poly)):
            return_list.append(poly[i] * i)
        return(return_list)
    else:
        return None
