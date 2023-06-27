#!/usr/bin/env python3
"""
Create a convolutional autoencoder
"""
import tensorflow.keras as keras


# Convolutional Autoencoder
def autoencoder(input_dims, filters, latent_dims):
    """
    Create a convolutional autoencoder

    Each convolution in the encoder and decoder should use a kernel size of
        (3, 3) with same padding and relu activation followed by max pooling
        size of (2, 2)

    Args:
        input_dims - tuple of integers containing the dimensions of the input
        filters - list containing the number of filters for each convolutional
            layer in the encoder, respectively
        The filters should be reversed for the decoder
        latent_dims - tuple of integers containing the dimensions of the latent
            space representation

    Returns:
        encoder - the encoder model
        decoder - the decoder model
        auto - the full autoencoder model
    """
    # i = 0
    # Build the encoder
    input_img = keras.Input(shape=input_dims)
    encoded = input_img
    for filter in filters:
        encoded = keras.layers.Conv2D(filter,
                                      (3, 3),
                                      activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2),
                                            padding='same')(encoded)
    encoder = keras.Model(input_img, encoded, name='encoder')

    # Build the decoder
    decoder_img = keras.Input(shape=latent_dims)
    decoded = decoder_img
    # print(filters[0])
    # print(filters[1])
    # print(filters[2])
    # Loop through filters backwards
    # for filter in filters[1::-1]:
    #     print("loop: " + str(i))
    #     print(filter)
    #     i += 1
    #     decoded = keras.layers.Conv2D(filter - 1,
    #                                   (3, 3),
    #                                   activation='relu',
    #                                   padding='same')(decoded)
    #     decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    # print(filters[0])
    # Going to try to manually loop
    decoded = keras.layers.Conv2D(filters[2],
                                  (3, 3),
                                  activation='relu',
                                  padding='same')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters[1],
                                  (3, 3),
                                  activation='relu',
                                  padding='same')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters[0],
                                  (3, 3),
                                  activation='relu',
                                  padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    # print("Exited loop")
    decoded = keras.layers.Conv2D(1,
                                  (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoded)
    decoder = keras.Model(decoder_img,
                          decoded,
                          name='decoder')

    # Bring it all together
    encoded_output = encoder(input_img)
    decoded_output = decoder(encoded_output)
    autoencoder = keras.Model(input_img,
                              decoded_output,
                              name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
