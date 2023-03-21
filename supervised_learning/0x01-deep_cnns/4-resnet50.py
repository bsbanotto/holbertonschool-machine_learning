#!/usr/bin/env python3
"""
Write a function that builds the ResNet-50 architecture as described in
    `Deep Residual Learning for Image Recognition (2015)`
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Assume input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by batch
    normalization along the channels axis and a ReLU activation
    All weights should use he normal initialization
    Returns the keras model
    """
    inputs = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=2,
                            padding='same',
                            kernel_initializer=init
                            )(inputs)

    batch_1 = K.layers.BatchNormalization(axis=3)(conv1)

    ReLU_1 = K.layers.ReLU()(batch_1)

    MaxPool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=2,
                                     padding='same'
                                     )(ReLU_1)

    conv2_a = projection_block(MaxPool1, [64, 64, 256], s=1)
    conv2_b = identity_block(conv2_a, [64, 64, 256])
    conv2_c = identity_block(conv2_b, [64, 64, 256])

    conv3_a = projection_block(conv2_c, [128, 128, 512])
    conv3_b = identity_block(conv3_a, [128, 128, 512])
    conv3_c = identity_block(conv3_b, [128, 128, 512])
    conv3_d = identity_block(conv3_c, [128, 128, 512])

    conv4_a = projection_block(conv3_d, [256, 256, 1024])
    conv4_b = identity_block(conv4_a, [256, 256, 1024])
    conv4_c = identity_block(conv4_b, [256, 256, 1024])
    conv4_d = identity_block(conv4_c, [256, 256, 1024])
    conv4_e = identity_block(conv4_d, [256, 256, 1024])
    conv4_f = identity_block(conv4_e, [256, 256, 1024])

    conv5_a = projection_block(conv4_f, [512, 512, 2048])
    conv5_b = identity_block(conv5_a, [512, 512, 2048])
    conv5_c = identity_block(conv5_b, [512, 512, 2048])

    AvgPool = K.layers.AveragePooling2D(pool_size=(1, 1),
                                        strides=1,
                                        padding='valid'
                                        )(conv5_c)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax'
                             )(AvgPool)

    return K.Model(inputs=inputs, outputs=softmax)
