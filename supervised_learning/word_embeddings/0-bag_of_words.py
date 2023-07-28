#!/usr/bin/env python3
"""
Bag of Words
"""
from sklearn.feature_extraction.text import CountVectorizer

import re
import string


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis

    Returns:
        embeddings: np.ndarray shape (s, f) containing the embeddings
            s: number of sentences in sentences
            f: number of features analyzed
        features: list of the features used for embeddings
    """
    # If vocab is None, make an empty list to append to
    if vocab is None:
        f = []
    else:
        f = vocab

    real_list = []
    corpus = []
    # First, make everything lowercase and remove punctuation
    for i in range(len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = review.split()
        review = ' '.join(review)
        corpus.append(review)

    for sentence in corpus:
        words = sentence.split()
        for word in words:
            if vocab is None and len(word) > 1 and word not in real_list:
                real_list.append(word)
                real_list.sort()

    cv = CountVectorizer()
    s = cv.fit_transform(corpus).toarray()
    f = real_list

    if vocab is None:
        return s, f
    return s, vocab
