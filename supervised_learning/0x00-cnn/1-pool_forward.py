#!/usr/bin/env python3
"""
Write a function that performs forward propagation over a pooling layer of a
neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
    output of the previous layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    kernel_shape: numpy.ndarray of shape (kh, kw) containing the kernels
    for the convolution
        kh: filter height
        kw: filter width
    stride: tuple of shape (sh, sw) containing the strides for the convolution
        sh: stride for the height
        sw: stride for the width
    mode: string containing either 'max' or 'avg', indicating whether to
    perform maximum or average pooling
    Returns the output of the pooling layer
    """
    m = A_prev.shape[0]
    h = A_prev.shape[1]
    w = A_prev.shape[2]
    c = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    h = (h - kh) // sh + 1
    w = (w - kw) // sw + 1

    output_image = np.zeros((m, h, w, c))

    for x in range(h):
        for y in range(w):
            i = x * sh
            j = y * sw
            if mode == 'max':
                output_image[:, x, y, :] = np.max(A_prev[:,
                                                         i:i + kh,
                                                         j:j + kw,
                                                         :], axis=(1, 2))
            if mode == 'avg':
                output_image[:, x, y, :] = np.mean(A_prev[:,
                                                          i:i + kh,
                                                          j:j + kw,
                                                          :], axis=(1, 2))
    return output_image
