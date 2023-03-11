#!/usr/bin/env python3
"""
Write a function that performs back propagation over a pooling layer of a
neural network
"""
import numpy as np
np.set_printoptions(suppress=True)


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the output of the pooling layer
        m: number of examples
        h_new: height of the output
        w_new: width of the output
        c: number of channels
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output
    of the previous layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c: number of channels
    kernel_shape: tuple of (kh, kw) containing the size of the kernel for
    pooling
        kh: kernel height
        kw: kernel width
    stride: tuple of (sh, sw) containing the strides for the pooling
        sh: stride for the height
        sw: stride for the width
    mode: string containing either 'max' or 'avg' indicating whether to
    perform maximum or average pooling, respectively
    Returns the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c_new = dA.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros_like(A_prev)

    for image in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    i = sh * x
                    j = sw * y
                    if mode == 'max':
                        a_prev_slice = A_prev[image, i:i + kh, j:j + kw, z]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[image, i:i + kh, j:j + kw, z] += (
                            mask * dA[image, x, y, z])
                    if mode == 'avg':
                        avgerage_dA = dA_prev[image, x, y, z] / kh / kw
                        dA_prev[image, i:i + kh, j:j + kw, z] += np.ones(
                            (kh, kw)) * avgerage_dA
    return dA_prev
