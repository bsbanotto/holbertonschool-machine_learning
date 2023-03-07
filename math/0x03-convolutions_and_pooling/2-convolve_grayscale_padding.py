#!/usr/bin/env python3
"""
Write a function that performs a convolution on grayscale images with custom
padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images: numpy.ndarray with shape (m, h, w) containing multiple images
        m: number of images
        h: height in pixels of the images
        w: width in pixels of the images
    kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
    convolution
        kh: height of the kernel
        kw: width of the kernel
    padding: tuple of(ph, pw)
        ph: padding for the height of the image
        pw: padding for the width of the image
        the image should be padded with 0's
    Only allowed to use two for loops
    Returns a numpy.ndarray containing the convolved images
    """

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    pad_top_bottom = padding[0]
    pad_left_right = padding[1]

    m = images.shape[0]
    h = images.shape[1] + (2 * padding[0]) - kh + 1
    w = images.shape[2] + (2 * padding[1]) - kw + 1

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right)))

    csm = np.zeros((m, h, w))

    for x in range(h):
        for y in range(w):
            hadamard_prod = np.multiply(images[:, x:x + kh, y:y + kw], kernel)
            csm[:, x, y] = np.sum(hadamard_prod, axis=(1, 2))
    return csm
