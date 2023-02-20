#!/usr/bin/env python3
"""
Implementation of the RMSProp algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function that updates a variable using the RMSProp algorithm
        Root Mean Square Propagation
    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid divide by zero error
    var: numpy.ndarray containing the variable to be updated(i.e. W or b)
    grad: numpy.ndarray containing the gradient of var(i.e. dW or db)
    s: previous second moment of var
    Returns the updated variable and new moment, respectively
        new_var will be passed as var in the next iteration
        new_moment will be passed as s in the next iteration
    """
    new_moment = beta2 * s + ((1 - beta2) * (grad ** 2))
    new_var = var - (alpha * (grad/((new_moment ** (1/2)) + epsilon)))

    return new_var, new_moment
