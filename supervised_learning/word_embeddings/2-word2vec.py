#!/usr/bin/env python3
"""
Module to create and train a word2vec model
"""
from gensim.models import Word2Vec


# Task 2. Train Word2Vec
def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Create and train a gensim word2vec model

    Args:
        sentences: List of sentences to be trained on
        size: Dimensionality of the embedding layer
        min_count: Minimum number of occurrences of a word for use in training
        window: Maximum distance between the current and predicted word within
            a sentence
        negative: Size of negative sampling
        cbow: Training type; True for CBOW, False for Skip-gram
        iterations: Number of iterations to train over
        seed: Seed for the random number generator
        workers: Number of worker threads to train the model

    Returns:
        model: The trained model
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1

    model = Word2Vec(sentences,
                     vector_size=size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     sg=sg,
                     negative=negative,
                     seed=seed,
                     epochs=iterations)

    return model
