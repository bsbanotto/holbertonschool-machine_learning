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

    short_sentences = []
    real_list = []
    corpus = []
    # First, convert sentences list to corpus, lowercase, no punctuation
    for i in range(len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = review.split()
        if vocab:
            short_sentences.clear()
            for j in range(0, len(review)):
                if review[j] in vocab:
                    word = review[j]
                    short_sentences.append(word)
            review = ' '.join(short_sentences)
        else:
            review = ' '.join(review)
        corpus.append(review)

    if vocab is None:
        print(corpus)
        # All of this to get the vocab list
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                if len(word) > 1 and word not in real_list:
                    real_list.append(word)
                    real_list.sort()
        cv = CountVectorizer()
        s = cv.fit_transform(corpus).toarray()
        f = real_list

        return s, f

    else:
        print(corpus)
        cv = CountVectorizer(vocabulary=vocab)
        s = cv.fit_transform(corpus).toarray()

        return s, vocab
