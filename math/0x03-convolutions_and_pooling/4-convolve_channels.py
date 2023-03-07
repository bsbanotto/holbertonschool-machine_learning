#!/usr/bin/env python3
"""
Write a function that performs a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images: numpy.ndarray with shape (m, h, w, c) containing multiple images
        m: number of images
        h: height in pixels of the images
        w: width in pixels of the images
        c: number of channels in the image
    kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel for
    the convolution
        kh: height of the kernel
        kw: width of the kernel
        c: number of channels in the kernel
    padding: tuple of(ph, pw)
        if `same`, performs a same convolution
        if `valid`, performs a valid convolution
        if a tuple (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
        the image should be padded with 0's
    stride: tuple of (sh, sw)
        sh: stride for the height of the image
        sw: stride for the width of the image
    Only allowed to use two for loops
    Returns a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    kc = kernel.shape[2]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        """Do same padding here"""
        pad_top_bottom = (((h - 1) * sh) + kh - h) // 2 + 1
        pad_left_right = (((w - 1) * sw) + kw - w) // 2 + 1

    elif padding == 'valid':
        """Do valid padding here"""
        pad_top_bottom = 0
        pad_left_right = 0

    else:
        """Do padding here"""
        pad_top_bottom = padding[0]
        pad_left_right = padding[1]

    h = (h + 2 * pad_top_bottom - kh) // sh + 1
    w = (w + 2 * pad_left_right - kw) // sw + 1

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right), (0, 0)))

    output_image = np.zeros((m, h, w))

    for x in range(h):
        for y in range(w):
            i = x * sh
            j = y * sw
            hadamard_prod = np.multiply(images[:, i:i + kh, j:j + kw], kernel)
            output_image[:, x, y] = np.sum(hadamard_prod, axis=(1, 2, 3))
    return output_image
