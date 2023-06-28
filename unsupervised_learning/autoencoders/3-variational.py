#!/usr/bin/env python3
"""
Implement a Variational Autoencoder
"""
import tensorflow.keras as keras


# Task 3 - Variational Autoencoder

def sampling(args):
    """
    Method to sample new, similar points from the latent space
    """
    z_mean, z_log_sigma, latent_dims = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)
                                          [0], latent_dims),
                                          mean=0.,
                                          stddev=0.1)

    return z_mean + keras.backend.exp(z_log_sigma) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder

    Args:
        input_dims - integer containing the dimension of the input
        hidden_layers - list containing the number of nodes for each hidden
            layer in the encoder, respectively
        The filters should be reversed for the decoder
        latent_dims - integer containing the dimension of the latent
            space representation

    Returns:
        encoder - the encoder model
        decoder - the decoder model
        auto - the full autoencoder model
    """
    # First, build the encoder
    input_img = keras.Input(shape=(input_dims,))
    h = input_img

    for layer in hidden_layers:
        h = keras.layers.Dense(layer, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma, latent_dims])

    encoder = keras.Model(input_img, [z_mean, z_log_sigma, z], name='encoder')

    # Now the decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for layer in hidden_layers[::-1]:
        x = keras.layers.Dense(layer, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate the VAE model
    outputs = decoder(encoder(input_img)[2])
    vae = keras.Model(input_img, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.binary_crossentropy(input_img,
                                                           outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)
    kl_loss -= keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae
