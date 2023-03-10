#!/usr/bin/env python3
"""
Write a function that performs forward propagation over a convolutional layer
of a neural network.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
    output of the previous layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
    for the convolution
        kh: filter height
        kw: filter width
        c_prev: number of channels in the previous layer
        c_new: number of channels in the output
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
    to the convolution
    activation: an activation function applied to the convolution
    padding: string that is either 'same' or 'valid' indicating the type of
    padding used
    stride: tuple of shape (sh, sw) containing the strides for the convolution
        sh: stride for the height
        sw: stride for the width
    Returns the output of the convolution layer
    """

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    kc_prev = W.shape[2]
    kc_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        pad_top_bottom = (((h_prev - 1) * sh) + kh - h_prev) // 2
        pad_left_right = (((w_prev - 1) * sw) + kw - w_prev) // 2

    if padding == 'valid':
        pad_top_bottom = 0
        pad_left_right = 0

    A_prev = np.pad(A_prev, ((0, 0), (pad_top_bottom, pad_top_bottom),
                    (pad_left_right, pad_left_right), (0, 0)))

    h_prev = (h_prev + 2 * pad_top_bottom - kh) // sh + 1
    w_prev = (w_prev + 2 * pad_left_right - kw) // sw + 1

    conv_image = np.zeros((m, h_prev, w_prev, kc_new))

    for x in range(h_prev):
        for y in range(w_prev):
            for z in range(kc_new):
                i = x * sh
                j = y * sw
                hadamard_prod = np.multiply(A_prev[:, i:i + kh, j:j + kw, :],
                                            W[:, :, :, z])
                conv_image[:, x, y, z] = np.sum(hadamard_prod, axis=(1, 2, 3))

    return activation(conv_image + b)
