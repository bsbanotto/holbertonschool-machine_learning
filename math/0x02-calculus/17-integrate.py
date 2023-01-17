#!/usr/bin/env python3
"""
This file determines the coefficients of a polynomial integral
"""


def poly_integral(poly, C=0):
    """
    Given a polynomial, determine the coefficients of the integral
    The integration constant is given as C
    """
    return_poly = []
    if(type(poly) == list) and len(poly) >= 0:
        return_poly.append(C)
        for i in range(0, len(poly)):
            append_value = poly[i] / (i + 1)
            if (float(append_value) == int(append_value)):
                append_value = int(append_value)
            return_poly.append(append_value)
    else:
        return None
    return (return_poly)
