#!/usr/bin/env python3
"""
Write a function that performs a convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
        if `same`, performs a same convolution
        if `valid`, performs a valid convolution
        if a tuple (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
        the image should be padded with 0's
    stride: tuple of sh, sw)
        sh: stride for the height of the image
        sw: stride for the width of the image
    Only allowed to use two for loops
    Returns a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        """Do same padding here"""
        h = h // sh
        w = w // sw
        pad_top_bottom = kh // 2
        pad_left_right = kw // 2

    elif padding == 'valid':
        """Do valid padding here"""
        h = (h - kh + 1) // sh
        w = (w - kw + 1) // sh
        pad_top_bottom = 0
        pad_left_right = 0

    else:
        """Do padding here"""
        h = (h + (2 * padding[0]) - kh + 1) // sh
        w = (w + (2 * padding[1]) - kw + 1) // sw
        pad_top_bottom = padding[0]
        pad_left_right = padding[1]

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right)))

    output_image = np.zeros((m, h, w))

    for x in range(h):
        for y in range(w):
            i = x * sh
            j = y * sw
            hadamard_prod = np.multiply(images[:, i:i + kh, j:j + kw], kernel)
            output_image[:, x, y] = np.sum(hadamard_prod, axis=(1, 2))
    return output_image
