#!/usr/bin/env python3
"""
Write a function that performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images: numpy.ndarray with shape (m, h, w, c) containing multiple images
        m: number of images
        h: height in pixels of the images
        w: width in pixels of the images
        c: number of channels in the image
    kernel_shape: numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution
        kh: height of the kernel
        kw: width of the kernel
    stride: tuple of (sh, sw)
        sh: stride for the height of the image
        sw: stride for the width of the image
    mode: indicates the type of pooling
        max: indicates max pooling
        avg: indicates average pooling
    Only allowed to use two for loops
    Returns a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    h = (h - kh) // sh + 1
    w = (w - kw) // sw + 1

    output_image = np.zeros((m, h, w, c))
    print(np.shape(output_image))

    for x in range(h):
        for y in range(w):
            i = x * sh
            j = y * sw
            if mode == 'max':
                output_image[:, x, y, :] = np.max(images[:,
                                                         i:i + kh,
                                                         j:j + kw,
                                                         :], axis=(1, 2))
            if mode == 'avg':
                output_image[:, x, y, :] = np.mean(images[:,
                                                          i:i + kh,
                                                          j:j + kw,
                                                          :], axis=(1, 2))
    return output_image
