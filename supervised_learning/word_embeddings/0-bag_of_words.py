#!/usr/bin/env python3
"""
Bag of Words
"""
import string
from sklearn.feature_extraction.text import CountVectorizer


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

    # First, make everything lowercase and remove punctuation
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('',
                                                    '',
                                                    string.punctuation))
        if vocab is None:
            for word in sentence.split():
                if word not in f:
                    f.append(word)
                # Sorted words to match test file output
                f.sort()

    cv = CountVectorizer(max_features=1500)
    s = cv.fit_transform(sentences).toarray()

    return s, f
