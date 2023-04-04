#!/usr/bin/env python3
"""
Transfer Learning Project
"""
# Imports
import os
import tensorflow.keras as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Data Preprocessing
def preprocess_data(X, Y):
    """
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR10 data
    Y: numpy.ndarray of shape (m,) containing the CIFAR10 data labels
    Returns X_p and Y_p
    """
    X_p = K.applications.efficientnet.preprocess_input(X, data_format=None)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


# Prep variables for use in model building / training
MODEL_PATH = 'cifar10.h5'
optimizer = K.optimizers.Adam()
init = K.initializers.he_normal()

# Load CIFAR10 dataset
(x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()

# Pre-process the data
x_train_p, y_train_p = preprocess_data(x_train, y_train)
x_valid_p, y_valid_p = preprocess_data(x_valid, y_valid)

# input tensor
inputs = K.Input(shape=(32, 32, 3))

# Resize input
resize = K.layers.Lambda(
    lambda image: K.backend.resize_images(
                                          image,
                                          240/32,
                                          240/32,
                                          data_format='channels_last'
                                          ))(inputs)

# Load pretrained EfficientNetB1 Base Bodel
base_model = K.applications.EfficientNetB1(
    include_top=False,
    weights='imagenet',
    input_tensor=resize,
    input_shape=(240, 240, 3)
)
base_model.trainable = False


def inception_block(A_prev, filters):
    """
    A_prev: output from the previous layer
    filters: tuple containing F1, F3R, F3, F5R, F5, FPP
        F1: number of filters in the 1x1 convolution
        F3R: number of filters in the 1x1 convolution before the 3x3
        convolution
        F3: number of filters in the 3x3 convolution
        F5R: number of filters in the 1x1 convolution before the 5x5
        convolution
        F5: number of filters in the 5x5 convolution
        FPP: number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block should use a ReLU activation
    Returns: the concatenated output of the inception block
    """
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filters[4]
    FPP = filters[5]

    conv_1x1 = K.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu'
                               )(A_prev)

    conv_1x1_3x3 = K.layers.Conv2D(filters=F3R,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   activation='relu'
                                   )(A_prev)

    conv_3x3 = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu'
                               )(conv_1x1_3x3)

    conv_1x1_5x5 = K.layers.Conv2D(filters=F5R,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   activation='relu'
                                   )(A_prev)

    conv_5x5 = K.layers.Conv2D(filters=F5,
                               kernel_size=(5, 5),
                               padding='same',
                               activation='relu'
                               )(conv_1x1_5x5)

    max_pool_3x3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                         strides=1,
                                         padding='same'
                                         )(A_prev)

    conv_1x1_pooled = K.layers.Conv2D(filters=FPP,
                                      kernel_size=(1, 1),
                                      padding='same',
                                      activation='relu'
                                      )(max_pool_3x3)

    output = K.layers.Concatenate()([
        conv_1x1, conv_3x3, conv_5x5, conv_1x1_pooled
    ])

    return output


# Add output layers
"""
Add the following layers:
    Inception Block
    MaxPooling
    Inception Block
    Global Average Pooling
    Softmax
"""
out = base_model.output
out = inception_block(out, [64, 96, 128, 16, 32, 32])
out = K.layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2,
                            padding='same')(out)
out = inception_block(out, [64, 96, 128, 16, 32, 32])
out = K.layers.GlobalAveragePooling2D()(out)
out = K.layers.Dense(10, activation='softmax')(out)

# Compile Model
model = K.models.Model(inputs=inputs, outputs=out)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(x=x_train_p,
          y=y_train_p,
          batch_size=64,
          epochs=5,
          validation_data=(x_valid_p, y_valid_p))

# Print Model Summary
model.summary()

# Save Model
model.save('cifar10.h5')
