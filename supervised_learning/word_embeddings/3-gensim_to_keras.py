#!/usr/bin/env python3
"""
Module to convert a word2vec model to a keras embedding layer
"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's
    learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being
        updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    Note: get_keras_embedding used to be a part of the KeyedVectors class in
    gensim.models. This was removed, and their wiki gave this function as a
    replacement. I copied this from their wiki, seems the same as using a
    library function.
    """
    # structure holding the result of training
    keyed_vectors = model.wv
    # vectors themselves, a 2D numpy array
    weights = keyed_vectors.vectors
    #  which row in `weights` corresponds to which word?
    index_to_key = keyed_vectors.index_to_key

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=False,
    )
    return layer
