#!/usr/bin/env python3
"""
Module that does a TF-IDF embedding
"""
from sklearn.feature_extraction.text import TfidfVectorizer

import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    Args:
        sentences: list of sentences to analyze
        vocab: list of vocab words to use for the analysis
            - If vocab is none, use all words within sentences

    Returns:
        embeddings: np.ndarray shape (s, f) containing the embeddings
            s: number of sentences in sentences
            f: number of features analyzed
        features: list of the features used for embeddings
    """
    if vocab:
        features = vocab
        vocab_sentences = []
    else:
        features = []

    corpus = []
    # Remove punctuation, to lowercase, create corpus
    for sentence in sentences:
        review = re.sub('[^a-zA-Z]', ' ', sentence)
        review = review.lower()
        review = review.split()
        # If vocab list is provided, need to remove all words except words that
        # are inside the vocab list.
        if vocab:
            vocab_sentences.clear()
            for j in range(0, len(review)):
                if review[j] in vocab:
                    word = review[j]
                    vocab_sentences.append(word)
            review = ' '.join(vocab_sentences)
        else:
            review = ' '.join(review)
        corpus.append(review)

    if vocab is None:
        # All of this to get the vocab list
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                if len(word) > 1 and word not in features:
                    features.append(word)
                    features.sort()

    tfidf = TfidfVectorizer(vocabulary=features)
    embedding = tfidf.fit_transform(corpus).toarray()

    return embedding, features
