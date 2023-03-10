#!/usr/bin/env python3
"""
Write a function that performs back propagation over a convolutional layer of
a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the unactivated output of the convolutional
    layer
        m: number of examples
        h_new: height of the output
        w_new: width of the output
        c_new: number of channels in the output
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
    stride: tuple of shape (sh, sw) containing the strides for the convolution
        sh: stride for the height
        sw: stride for the width
    Returns the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW) and the biases(db), respectively
    """
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2))

    if padding == 'same':
        pad_top_bottom = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pad_left_right = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    if padding == 'valid':
        pad_top_bottom = 0
        pad_left_right = 0

    A_prev = np.pad(A_prev, ((0, 0), (pad_top_bottom, pad_top_bottom),
                    (pad_left_right, pad_left_right), (0, 0)))

    dA_prev = np.pad(dA_prev, ((0, 0), (pad_top_bottom, pad_top_bottom),
                     (pad_left_right, pad_left_right), (0, 0)))

    for image in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    i = x * sh
                    j = y * sw
                    dW[:, :, :, z] +=\
                        np.multiply(A_prev[image, i:i + kh, j:j + kw, :],
                                    dZ[image, x, y, z])
                    dA_prev[image, i:i + kh, j:j + kw, :] +=\
                        np.multiply(W[:, :, :, z], dZ[image, x, y, z])

    if padding == 'same':
        dA_prev = dA_prev[:, pad_top_bottom:-pad_top_bottom,
                          pad_left_right:-pad_left_right, :]

    return dA_prev, dW, db
