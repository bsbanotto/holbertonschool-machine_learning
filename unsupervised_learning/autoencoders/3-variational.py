#!/usr/bin/env python3
import tensorflow.keras as keras
layers = keras.layers
Model = keras.Model


def sampling(args):
    """
    Sampling method
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    autoencoder method
    """
    encoder_inputs = layers.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
    z_mean = layers.Dense(latent_dims)(x)
    z_log_var = layers.Dense(latent_dims)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z, z_mean, z_log_var])

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
    decoder_outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(latent_inputs, decoder_outputs)

    # Full Autoencoder
    autoencoder_outputs = decoder(encoder(encoder_inputs)[0])
    autoencoder = Model(encoder_inputs, autoencoder_outputs)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
