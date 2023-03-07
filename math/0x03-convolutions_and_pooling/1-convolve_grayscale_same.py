#!/usr/bin/env python3
"""
Function that performs a same convolution on greyscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images: numpy.nparray with shape (m, h, w) containing multiple greyscale
    images
        m: number of images
        h: height of the images in pixels
        w: width of the images in pixels
    kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
    convolution
        kh: kernel height
        kw: kernel width
    If necessary, the image should be padded with 0's
    Only allowed to use two for loops, no loops of other kind
    Returns a numpy.ndarray containing the convolved images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    pad_top_bottom = kh // 2
    pad_left_right = kw // 2

    images = np.pad(images, ((0, 0), (pad_top_bottom, pad_top_bottom),
                             (pad_left_right, pad_left_right)))
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    csm = np.zeros((m, h - pad_top_bottom * 2, w - pad_left_right * 2))

    for x in range(1, (h - 2)):
        for y in range(1, (w - 2)):
            hadamard_prod = np.multiply(images[:, x:x + kh, y:y + kw], kernel)
            csm[:, x, y] = np.sum(hadamard_prod, axis=(1, 2))
    return csm
