#!/usr/bin/env python3
"""
Implementation of Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Implementation of Adam optimization algorithm
    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid divide by zero error
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: previous first moment of var
    s: previous second moment of var
    t: time step used for bias correction
    Returns the updated variable, new first moment and new second moment
        new first moment = Vd
        new second moment = Sd
        updated variable = new_var
    """
    Vd = 0
    Sd = 0

    Vd = beta1 * v + ((1 - beta1) * grad)
    Sd = beta2 * s + ((1 - beta2) * (grad ** 2))
    Vd_correct = Vd / (1 - beta1 ** t)
    Sd_correct = Sd / (1 - beta2 ** t)
    new_var = var - (alpha * (Vd_correct / ((Sd_correct ** (1/2)) + epsilon)))

    return new_var, Vd, Sd
