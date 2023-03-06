#!/usr/bin/env python3
"""
Function that performs a valid convolution on greyscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    Only allowed to use two for loops, no loops of other kind
    Returns a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    cvh = h - kh + 1  # convolve valid height
    cvw = w - kw + 1  # convolve valid width
    cvm = np.zeros((m, cvh, cvw))  # convolve valid matrix

    for x in range(cvh):
        for y in range(cvw):
            hadamard_prod = np.multiply(images[:, x:x + kh, y:y + kw], kernel)
            cvm[:, x, y] = np.sum(hadamard_prod, axis=(1, 2))
    return cvm
