#!/usr/bin/env python3
"""
Updates a variable using gradient descent with momentum optimization
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum optimization
    alpha: learning rate
    beta1: momentum weight
    var: numpy.ndarray containing the variable to be updated(i.e. W or b)
    grad: numpy.ndarray containing the gradient of var(i.e. dW or db)
    v: previous first moment of var(dW_prev or db_prev)
    Returns the updated variable and the new moment
    """
    new_var = beta1 * v + (grad * (1 - beta1))
    new_v = var - (alpha * new_var)

    return new_v, new_var
