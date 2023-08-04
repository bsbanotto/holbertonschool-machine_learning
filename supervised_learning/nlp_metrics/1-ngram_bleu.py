#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score for a sentence
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Args:
        references: list of reference translations
            - Each reference translataion is a list of the words in the
              translation
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns:
        the unigram BLEU score

    Notes:
        Similar to unigram, except need to make a dictionaries of tuples to
        find matches.
    """
    # Calculate n-gram counts in the sentence (Create corpus of tuples)
    corpus = {}
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i + n])
        corpus[ngram] = corpus.get(ngram, 0) + 1
    w_t = len(corpus)

    # Calculate maximum n-gram counts in the references
    max_counts = {}
    for reference in references:
        ref_counts = {}
        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i + n])
            ref_counts[ngram] = ref_counts.get(ngram, 0) + 1
        for ngram, count in ref_counts.items():
            max_counts[ngram] = max(max_counts.get(ngram, 0), count)

    # Calculate clipped n-gram counts
    clipped_counts = {}
    for ngram, count in corpus.items():
        clipped_counts[ngram] = min(count, max_counts.get(ngram, 0))
    m = sum(clipped_counts.values())

    P = m / w_t

    # Calculate brevity penalty
    ref_len = min(len(reference) for reference in references)
    c = len(sentence)
    BP = min(1, np.exp(1 - ref_len / c))

    return P * BP
